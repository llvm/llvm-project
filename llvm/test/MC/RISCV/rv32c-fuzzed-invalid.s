# REQUIRES: asserts
# RUN: llvm-mc --verify -triple riscv32 -mattr=+c,+f,+d < %s
#
# Fuzzed test cases produced by a LLVM MC Assembler
# Protocol Buffer Fuzzer for the RISC-V assembly language.
#

c.addi x13,f30,0
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:8 {{register must be zero (x0)}}
# expected-note@-3:12 {{immediate must be an integer in the range [-32, 31]}}
c.swsp x0,(f14)     # expected-error:12 {{register must be sp (x2)}}
c.lui x4,x0         # expected-error:10 {{immediate must be in [0xfffe0, 0xfffff] or [1, 31]}}
c.li x6,x6,x0,x0    # expected-error:9 {{immediate must be an integer in the range [-32, 31]}}
c.addi16sp 2,(x0)   # expected-error:12 {{register must be sp (x2)}}
c.fsdsp f9,x0,0     # expected-error:12 {{expected '('}}
c.flw f15,x14,x0    # expected-error:11 {{expected '('}}
c.fld f8,f30,x17    # expected-error:10 {{expected '('}}
c.addi4spn x8,x2,x8 # expected-error:18 {{immediate must be a multiple of 4 bytes in the range [4, 1020]}}
