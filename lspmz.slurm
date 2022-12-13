#!/bin/bash
#SBATCH --time=09:15:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=64g
#SBATCH --job-name=lsp
#SBATCH --partition=gpu
#SBATCH --gres=gpu

source activate lsp
python lspmz.py
