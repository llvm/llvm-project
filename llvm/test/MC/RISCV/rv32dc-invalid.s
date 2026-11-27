# RUN: llvm-mc --verify -triple=riscv32 -mattr=+c,+d < %s

## FPRC
c.fld  ft3, 8(a5) # expected-error:8 {{register must be a FPR}}

## uimm9_lsb000
c.fldsp  fs1, 512(sp) # expected-error:15 {{immediate must be a multiple of 8 bytes in the range [0, 504]}}
c.fsdsp  fs2, -8(sp) # expected-error:15 {{immediate must be a multiple of 8 bytes in the range [0, 504]}}

## uimm8_lsb000
c.fld  fs0, -8(sp) # expected-error:13 {{immediate must be a multiple of 8 bytes in the range [0, 248]}}
c.fsd  fs1, 256(sp) # expected-error:13 {{immediate must be a multiple of 8 bytes in the range [0, 248]}}
