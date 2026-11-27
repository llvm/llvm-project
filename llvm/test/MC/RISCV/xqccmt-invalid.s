# RUN: llvm-mc --verify -triple=riscv32 -mattr=+experimental-xqccmt -M no-aliases -show-encoding < %s
# RUN: llvm-mc --verify -triple=riscv64 -mattr=+experimental-xqccmt -M no-aliases -show-encoding < %s

qc.cm.jt 32 # expected-error {{immediate must be an integer in the range [0, 31]}}

qc.cm.jalt 256 # expected-error {{immediate must be an integer in the range [32, 255]}}

qc.cm.jalt 31 # expected-error {{immediate must be an integer in the range [32, 255]}}
