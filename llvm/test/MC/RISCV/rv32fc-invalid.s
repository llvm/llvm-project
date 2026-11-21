# RUN: llvm-mc --verify -triple=riscv32 -mattr=+c,+f < %s

## FPRC
c.flw  ft3, 8(a5) # expected-error:8 {{register must be a FPR from f8 to f15 (fs0-fs1, fa0-fa5)}}

## uimm8_lsb00_optional
c.flwsp  fs1, 256(sp) # expected-error:15 {{immediate must be a multiple of 4 bytes in the range [0, 252]}}
c.fswsp  fs2, -4(sp) # expected-error:15 {{immediate must be a multiple of 4 bytes in the range [0, 252]}}

## uimm7_lsb00
c.flw  fs0, -4(sp) # expected-error:13 {{immediate must be a multiple of 4 bytes in the range [0, 124]}}
c.fsw  fs1, 128(sp) # expected-error:13 {{immediate must be a multiple of 4 bytes in the range [0, 124]}}
