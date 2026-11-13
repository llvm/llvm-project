# With Zks extension:
# RUN: llvm-mc --verify -triple=riscv32 -mattr=+zks < %s

# With Zksed extension:
# RUN: llvm-mc --verify -triple=riscv32 -mattr=+zksed < %s

sm4ed a0, a1, a2, 8 # expected-error {{immediate must be an integer in the range [0, 3]}}

sm4ks a0, a1, a2, 8 # expected-error {{immediate must be an integer in the range [0, 3]}}
