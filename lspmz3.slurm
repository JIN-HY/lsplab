#!/bin/bash
#SBATCH --time=09:15:00
#SBATCH --mem-per-cpu=500g
#SBATCH --job-name=lsp
#SBATCH --partition=batch


source activate lsp
python lspmz.py
