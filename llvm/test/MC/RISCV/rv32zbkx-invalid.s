# RUN: llvm-mc --verify -triple riscv32 -mattr=+zbkx < %s

# Too few operands
xperm8 t0, t1 # expected-error:14 {{too few operands for instruction}}
# Too few operands
xperm4 t0, t1 # expected-error:14 {{too few operands for instruction}}
