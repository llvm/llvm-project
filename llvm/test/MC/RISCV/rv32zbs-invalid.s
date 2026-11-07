# RUN: llvm-mc --verify -triple riscv32 -mattr=+zbs < %s

# Too few operands
bclr t0, t1 # expected-error:12 {{too few operands for instruction}}
# Too few operands
bset t0, t1 # expected-error:12 {{too few operands for instruction}}
# Too few operands
binv t0, t1 # expected-error:12 {{too few operands for instruction}}
# Too few operands
bext t0, t1 # expected-error:12 {{too few operands for instruction}}
# Too few operands
bclri t0, t1 # expected-error:13 {{too few operands for instruction}}
# Immediate operand out of range
bclri t0, t1, 32 # expected-error:15 {{immediate must be an integer in the range [0, 31]}}
bclri t0, t1, -1 # expected-error:15 {{immediate must be an integer in the range [0, 31]}}
# Too few operands
bseti t0, t1 # expected-error:13 {{too few operands for instruction}}
# Immediate operand out of range
bseti t0, t1, 32 # expected-error:15 {{immediate must be an integer in the range [0, 31]}}
bseti t0, t1, -1 # expected-error:15 {{immediate must be an integer in the range [0, 31]}}
# Too few operands
binvi t0, t1 # expected-error:13 {{too few operands for instruction}}
# Immediate operand out of range
binvi t0, t1, 32 # expected-error:15 {{immediate must be an integer in the range [0, 31]}}
binvi t0, t1, -1 # expected-error:15 {{immediate must be an integer in the range [0, 31]}}
# Too few operands
bexti t0, t1 # expected-error:13 {{too few operands for instruction}}
# Immediate operand out of range
bexti t0, t1, 32 # expected-error:15 {{immediate must be an integer in the range [0, 31]}}
bexti t0, t1, -1 # expected-error:15 {{immediate must be an integer in the range [0, 31]}}
