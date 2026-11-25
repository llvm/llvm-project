# RUN: llvm-mc --verify -triple riscv32 -mattr=+zimop < %s

# Too few operands
mop.r.0 t0 # expected-error:11 {{too few operands for instruction}}
# Too few operands
mop.rr.0 t0, t1 # expected-error:16 {{too few operands for instruction}}
