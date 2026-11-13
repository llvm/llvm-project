# RUN: llvm-mc --verify -triple riscv32 -mattr=+zbkc < %s

# Too few operands
clmul t0, t1 # expected-error:13 {{too few operands for instruction}}
# Too few operands
clmulh t0, t1 # expected-error:14 {{too few operands for instruction}}

# Undefined zbc instruction in zbkc
clmulr t0, t1, t2 # expected-error:1 {{instruction requires the following: 'Zbc' (Carry-Less Multiplication)}}
