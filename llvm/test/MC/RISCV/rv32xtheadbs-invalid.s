# RUN: llvm-mc --verify -triple riscv32 -mattr=+xtheadbs < %s

# Too few operands
th.tst t0, t1 # expected-error:14 {{too few operands for instruction}}
# Immediate operand out of range
th.tst t0, t1, 32 # expected-error:16 {{immediate must be an integer in the range [0, 31]}}
th.tst t0, t1, -1 # expected-error:16 {{immediate must be an integer in the range [0, 31]}}
