# RUN: llvm-mc --verify -triple riscv64 -mattr=+zba < %s

# Too few operands
slli.uw t0, t1 # expected-error:15 {{too few operands for instruction}}
# Immediate operand out of range
slli.uw t0, t1, 64 # expected-error:17 {{immediate must be an integer in the range [0, 63]}}
slli.uw t0, t1, -1 # expected-error:17 {{immediate must be an integer in the range [0, 63]}}
# Too few operands
add.uw t0, t1 # expected-error:14 {{too few operands for instruction}}
# Too few operands
sh1add.uw t0, t1 # expected-error:17 {{too few operands for instruction}}
# Too few operands
sh2add.uw t0, t1 # expected-error:17 {{too few operands for instruction}}
# Too few operands
sh3add.uw t0, t1 # expected-error:17 {{too few operands for instruction}}
