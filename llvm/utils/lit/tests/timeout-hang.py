# REQUIRES: lit-max-individual-test-time

# https://github.com/llvm/llvm-project/issues/133914

# RUN: not %{lit} %{inputs}/timeout-hang/run-nonexistent.py \
# RUN: --timeout=1 --param external=0 -a
