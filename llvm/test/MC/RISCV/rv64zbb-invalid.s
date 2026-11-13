# RUN: llvm-mc --verify -triple riscv64 -mattr=+zbb < %s

# Too many operands
clzw t0, t1, t2 # expected-error:14 {{unexpected extra operand for instruction}}
# Too many operands
ctzw t0, t1, t2 # expected-error:14 {{unexpected extra operand for instruction}}
# Too many operands
cpopw t0, t1, t2 # expected-error:15 {{unexpected extra operand for instruction}}
# Too few operands
rolw t0, t1 # expected-error:12 {{too few operands for instruction}}
# Too few operands
rorw t0, t1 # expected-error:12 {{too few operands for instruction}}
# Too few operands
roriw t0, t1 # expected-error:13 {{too few operands for instruction}}
# Immediate operand out of range
roriw t0, t1, 32 # expected-error:15 {{immediate must be an integer in the range [0, 31]}}
roriw t0, t1, -1 # expected-error:15 {{immediate must be an integer in the range [0, 31]}}
rori t0, t1, 64 # expected-error:14 {{immediate must be an integer in the range [0, 63]}}
rori t0, t1, -1 # expected-error:14 {{immediate must be an integer in the range [0, 63]}}
