# RUN: llvm-mc --verify -triple riscv32 -mattr=+zbc < %s

# Too few operands
clmul t0, t1 # expected-error:13 {{too few operands for instruction}}
# Too few operands
clmulr t0, t1 # expected-error:14 {{too few operands for instruction}}
# Too few operands
clmulh t0, t1 # expected-error:14 {{too few operands for instruction}}
