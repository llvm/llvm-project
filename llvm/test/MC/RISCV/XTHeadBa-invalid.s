# RUN: llvm-mc --verify -triple riscv32 -mattr=+xtheadba < %s
# RUN: llvm-mc --verify -triple riscv64 -mattr=+xtheadba < %s

# Too few operands
th.addsl t0, t1 # expected-error:16 {{too few operands for instruction}}
# Too few operands
th.addsl t0, t1, t2 # expected-error:20 {{too few operands for instruction}}
# Immediate operand out of range
th.addsl t0, t1, t2, 4 # expected-error:22 {{immediate must be an integer in the range [0, 3]}}
# Immediate operand out of range
th.addsl t0, t1, t2, -1 # expected-error:22 {{immediate must be an integer in the range [0, 3]}}
