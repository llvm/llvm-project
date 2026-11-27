# With Zk extension:
# RUN: llvm-mc --verify -triple=riscv64 -mattr=+zk < %s

# With Zkn extension:
# RUN: llvm-mc --verify -triple=riscv64 -mattr=+zkn < %s

# With Zkne extension:
# RUN: llvm-mc --verify -triple=riscv64 -mattr=+zkne < %s

aes64ks1i a0, a1, 11 # expected-error {{immediate must be an integer in the range [0, 10]}}

aes64ks1i a0, a1, -1 # expected-error {{immediate must be an integer in the range [0, 10]}}
