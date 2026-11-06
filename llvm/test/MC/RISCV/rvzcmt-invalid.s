# RUN: llvm-mc --verify -triple=riscv32 -mattr=+zcmt -M no-aliases -show-encoding < %s
# RUN: llvm-mc --verify -triple=riscv64 -mattr=+zcmt -M no-aliases -show-encoding < %s

# expected-error@+1 {{immediate must be an integer in the range [0, 31]}}
cm.jt 64

# expected-error@+1 {{immediate must be an integer in the range [32, 255]}}
cm.jalt 256

# expected-error@+1 {{immediate must be an integer in the range [32, 255]}}
cm.jalt 31
