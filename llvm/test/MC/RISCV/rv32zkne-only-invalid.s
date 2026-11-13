# With Zk extension:
# RUN: llvm-mc --verify -triple=riscv32 -mattr=+zk < %s

# With Zkn extension:
# RUN: llvm-mc --verify -triple=riscv32 -mattr=+zkn < %s

# With Zkne extension:
# RUN: llvm-mc --verify -triple=riscv32 -mattr=+zkne < %s

aes32esmi a0, a1, a2, 8 # expected-error {{immediate must be an integer in the range [0, 3]}}

aes32esi a0, a1, a2, 8 # expected-error {{immediate must be an integer in the range [0, 3]}}
