# With Zk extension:
# RUN: llvm-mc --verify -triple=riscv32 -mattr=+zk < %s

# With Zkn extension:
# RUN: llvm-mc --verify -triple=riscv32 -mattr=+zkn < %s

# With Zknd extension:
# RUN: llvm-mc --verify -triple=riscv32 -mattr=+zknd < %s

aes32dsmi a0, a1, a2, 8 # expected-error {{immediate must be an integer in the range [0, 3]}}

aes32dsi a0, a1, a2, 8 # expected-error {{immediate must be an integer in the range [0, 3]}}
