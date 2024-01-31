import tqdm
import torch
import torch.nn as nn
import json
import numpy as np
import time
import random
import argparse
import torch
import matplotlib.pyplot as plt
from torch import optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tokenizer import get_tokenizer
import os
from model_tfmr import TfmrLMHeadModel, TransposeLinear
from torch.profiler import ExecutionTraceObserver

from configuration import ModelConfig

gpu_id = "0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, default="run",
    help="Experiment name. Default: run")
parser.add_argument("--model_config", type=str, default="./config.json",
    help="Path to the configuration file. Default: ./config.json")    
parser.add_argument("--tokenizer_dir", type=str, default="./tokenizer",
    help="Tokenizer file directory. Default: ./tokenizer")    
parser.add_argument("--num_epochs", type=int, default=20,
    help="Number of training epoch. Default: 20")
parser.add_argument("--cpu_count", type=int, default=20,
    help="Number of CPU cores for evaluation. Default: 20")    
parser.add_argument("--batch_size", type=int, default=32,
    help="The number of batch_size. Default: 32")
parser.add_argument("--learning_rate", type=float, default=1e-4,
    help="Learning rate during optimization. Default: 1e-4")
parser.add_argument("--test", type=str, default=None,
    help="Evaluate the model with the specified name. Default: None")
parser.add_argument("--data_dir", type=str, default="./data",
    help="Data directory. Default: ../data")
parser.add_argument("--train_dir", type=str, default="./train_test",
    help="Training directory for saving model. Default: ./train")
parser.add_argument("--pretrain_dir", type=str, default=None,
    help="Pre-Training directory for loading pretrained model. Default: None")
parser.add_argument("--maxlen", type=int, default=35,
    help="Maximum length for training/inference. Default: 35")    
parser.add_argument("--decode_strategy", type=str, choices=["random", "top-p"], default="random",
    help="The strategy for decoding. Can be \"random\" or \"top-p\". Default: random")
parser.add_argument("--temperature", type=float, default=1,
    help="The temperature for decoding. Default: 1")
parser.add_argument("--top_p", type=float, default=1.0,
    help="The p for top-p sampling. Default: 1.0")
args = parser.parse_args()
local_rank=int(os.environ['RANK'])


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
def _sentence_bleu(ele):
    return sentence_bleu(ele[0], ele[1], weights=ele[2], smoothing_function=SmoothingFunction().method1)


def fast_evaluate(model, data, batch_size, PAD_ID, device):
    model.eval()
    st, ed, all_loss = 0, 0, []
    while ed < len(data):
        st, ed = ed, (ed + batch_size) if (ed + batch_size < len(data)) else len(data)
        with torch.no_grad():
            input_ids = torch.tensor(data[st:ed]).to(device)
            ce_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

            outputs = model(input_ids)
            lm_logits = outputs["logits"]
            shift_logits = lm_logits[ : , : -1, : ].contiguous()
            shift_labels = input_ids[ : , 1 : ].contiguous()
            element_loss = ce_loss_fct(shift_logits.transpose(1, 2), shift_labels)
            mask = torch.ones_like(input_ids, dtype = torch.float)
            mask[ : , 1 : ] = (shift_labels != PAD_ID).float()
            mask = mask[ : , : -1]
            loss = (element_loss * mask).sum(dim = 1) / mask.sum(dim = 1)
            all_loss.append(loss)
    loss = torch.mean(torch.cat(all_loss, dim=0), dim=0)
    ppl = torch.exp(loss).item()
    model.train()
    return loss, ppl


def evaluate(gen_ids, truth_ids, cpu_count=20):
    from multiprocessing import Pool

    assert len(gen_ids) == len(truth_ids)
    sample_hyps_num = len(gen_ids)
    res = {}
    for ngrams in [4]:
        print(f"computing BLEU-{ngrams}")
        bleu_irl_fw, bleu_irl_bw = [], []
        weights = np.ones(ngrams) / ngrams

        tasks = ((truth_ids, gen_ids[i], weights) for i in range(sample_hyps_num))
        pool = Pool(cpu_count)
        values = pool.imap_unordered(_sentence_bleu, tasks, chunksize=20)
        values = tqdm.tqdm(values, total=sample_hyps_num)
        for ans in values:
            bleu_irl_fw.append(ans)
        pool.close()
        pool.join()

        tasks = ((gen_ids, truth_ids[i], weights) for i in range(sample_hyps_num))
        pool = Pool(cpu_count)
        values = pool.imap_unordered(_sentence_bleu, tasks, chunksize=20)
        values = tqdm.tqdm(values, total=sample_hyps_num)
        for ans in values:
            bleu_irl_bw.append(ans)
        pool.close()
        pool.join()

        fw_bleu = (1.0 * sum(bleu_irl_fw) / len(bleu_irl_fw))
        bw_bleu = (1.0 * sum(bleu_irl_bw) / len(bleu_irl_bw))
        if fw_bleu + bw_bleu > 0:
            fw_bw_bleu = 2.0 * bw_bleu * fw_bleu / (fw_bleu + bw_bleu)
        else:
            fw_bw_bleu = 0

        res.update({f"fw-bleu-{ngrams}" : fw_bleu, \
            f"bw-bleu-{ngrams}" : bw_bleu, \
            f"fw-bw-bleu-{ngrams}" : fw_bw_bleu \
        })
    return res


def load_data(path, tokenizer, PAD_ID, field_list=["train", "dev", "test"], maxlen=40):
    data, data_remove_pad = {}, {}
    for name in field_list:
        data[name], data_remove_pad[name] = [], []
        with open(f"{path}/{name}.txt") as fin:
            for line in fin:
                tokens = tokenizer.encode(line.strip())
                if len(tokens) < maxlen:
                    data[name].append([PAD_ID] + tokens + [PAD_ID]*(maxlen - len(tokens)))
                else:
                    data[name].append([PAD_ID] + tokens[:maxlen])
                data_remove_pad[name].append(tokens)
    return data, data_remove_pad


def load_model(pretrained_dir, model_name="pretrained_ckpt.bin"):
    model_path = os.path.join(pretrained_dir, model_name)

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        config_path = os.path.join(pretrained_dir, "config.json")
        assert os.path.exists(config_path), f"config.json must exist in {pretrained_dir}"
        print(f"[WARNING] Overiding args.model_config with {config_path}")
        with open(config_path) as f:
            model_config = json.load(f)
            model_config = ModelConfig(**model_config)
        model = TfmrLMHeadModel(model_config)
        model.load_state_dict(torch.load(model_path))
    else:
        raise RuntimeError(f"No such checkpoint: {model_path}")
    
    return model, model_config


def get_init_weights_func(config):
    def init_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, TransposeLinear)):
            module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    return init_weights

def trace_handler(prof):
    prof.export_chrome_trace(f'./kineto_trace{local_rank}.json')

def main():
    print(args)
    set_seed(1229)
    device = torch.device('cuda', local_rank)
    if not os.path.exists(args.train_dir):
        os.mkdir(args.train_dir)
    tokenizer = get_tokenizer(args.tokenizer_dir)
    PAD_ID = tokenizer.encoder["<|endoftext|>"]
    print("Tokenizer PAD ID:", PAD_ID)

    print("Loading Data ...")
    data, data_remove_pad = load_data(path=args.data_dir, tokenizer=tokenizer, PAD_ID=PAD_ID, field_list=["train", "dev", "test"], maxlen=args.maxlen)
    if args.test is None:
        if args.pretrain_dir is None:
            print("Created model with fresh parameters.")
            with open(args.model_config) as fin:
                model_config = json.load(fin)
                config = ModelConfig(**model_config)
            model = TfmrLMHeadModel(config)
            init_weights_func = get_init_weights_func(config=config)
            model.apply(init_weights_func)
        else:
            model, config = load_model(args.pretrain_dir)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        model.to(device)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if local_rank==0:
            print(model)
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)
        best_val_ppl = float("inf")
        best_epoch = -1

        train_dataset = torch.tensor(data["train"])
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=os.environ['WORLD_SIZE'], sampler=train_sampler, pin_memory=False)

        eg = None
        eg_file = f"eg{local_rank}.json"
        eg = ExecutionTraceObserver()
        eg.register_callback(eg_file)

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=0,
                warmup=5,
                active=1,
            ),
            record_shapes=True,
            on_trace_ready=trace_handler,
        ) as prof:
            for epoch in range(1, args.num_epochs + 1):
                if epoch == 6:
                    eg.start()
                start_time = time.time()
                batch_num = 0
                losses = []
                train_loader.sampler.set_epoch(epoch)
                for train_data in train_loader:
                    batch_num += 1
                    batched_data = torch.as_tensor(train_data).to(device)
                    optimizer.zero_grad()
                    loss = model(input_ids=batched_data, labels=batched_data, PAD_ID=PAD_ID)["loss"]
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.tolist())

                train_loss = np.mean(losses)

                val_loss, val_ppl = fast_evaluate(model=model, data=data["dev"], batch_size=args.batch_size, PAD_ID=PAD_ID, device=device)
                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    best_epoch = epoch

                    torch.save(model.state_dict(), os.path.join(args.train_dir, f"checkpoint_{args.name}.bin"))
                    with open(os.path.join(args.train_dir, "config.json"), "w") as f:
                        json.dump(config.__dict__, f)

                    epoch_time = time.time() - start_time
                    if local_rank==0:
                        print("Epoch " + str(epoch) + " of " + str(args.num_epochs) + " took " + str(epoch_time) + "s")
                        print("  training loss:                 " + str(train_loss))
                        print("  validation loss:               " + str(val_loss))
                        print("  validation perplexity:         " + str(val_ppl))
                        print("  best epoch:                    " + str(best_epoch))
                        print("  best validation perplexity:    " + str(best_val_ppl))
                else:
                    if local_rank==0:
                        print("Validation perplexity: {:.3f}, becomes larger. Stop training.".format(val_ppl))
                    break
                if epoch == 6:
                    eg.stop()
                    eg.unregister_callback()
                prof.step()

    else:
        model, config = load_model(args.train_dir, model_name=f"checkpoint_{args.test}.bin")
        model.to(device)
        print(model)
        test_loss, test_ppl = fast_evaluate(model=model, data=data["test"], batch_size=args.batch_size, PAD_ID=PAD_ID, device=device)
        print("        test_set, perplexity {:.2f}".format(test_ppl))
        result = model.inference(device=device, PAD_ID=PAD_ID, 
            batch_size=args.batch_size, maxlen=args.maxlen, decode_strategy=args.decode_strategy, temperature=args.temperature, top_p=args.top_p)
        with open(f"output_{args.decode_strategy}.txt", "w") as fout:
            for k, output in enumerate(result):
                out = tokenizer.decode(output)
                # print(k, out)
                fout.write(out + "\n")
        eval_result = evaluate(gen_ids=result, truth_ids=data_remove_pad["test"])
        print("        test_set, forward BLEU-4 {:.3f}, backward BLEU-4 {:.3f}, harmonic BLEU-4 {:.3f}".format(eval_result["fw-bleu-4"], eval_result["bw-bleu-4"], eval_result["fw-bw-bleu-4"]))
        print(f"        test_set, write inference results to output_{args.decode_strategy}.txt")


if __name__ == "__main__":
    main()