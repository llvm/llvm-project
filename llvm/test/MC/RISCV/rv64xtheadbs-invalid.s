# RUN: llvm-mc --verify -triple riscv64 -mattr=+xtheadbs < %s

# Immediate operand out of range
th.tst t0, t1, 64 # expected-error:16 {{immediate must be an integer in the range [0, 63]}}
th.tst t0, t1, -1 # expected-error:16 {{immediate must be an integer in the range [0, 63]}}
