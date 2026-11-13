# RUN: llvm-mc --verify -triple=riscv32 -mattr=zcb -M no-aliases -show-encoding %s
# RUN: llvm-mc --verify -triple=riscv64 -mattr=zcb -M no-aliases -show-encoding < %s

c.lbu a5, 10(a4) # expected-error {{immediate must be an integer in the range [0, 3]}}

c.lhu a5, 10(a4) # expected-error {{immediate must be one of [0, 2]}}

c.lh a5, 10(a4) # expected-error {{immediate must be one of [0, 2]}}

c.sb a5, 10(a4) # expected-error {{immediate must be an integer in the range [0, 3]}}

c.sh a5, 10(a4) # expected-error {{immediate must be one of [0, 2]}}
