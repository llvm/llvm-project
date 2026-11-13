# RUN: llvm-mc --verify %s -triple=riscv32 -mattr=+zmmul -M no-aliases

div s0, s0, s0 # expected-error {{instruction requires the following: 'M' (Integer Multiplication and Division)}}

divu gp, a0, a1 # expected-error {{instruction requires the following: 'M' (Integer Multiplication and Division)}}

rem s2, s2, s8 # expected-error {{instruction requires the following: 'M' (Integer Multiplication and Division)}}

remu x18, x18, x24 # expected-error {{instruction requires the following: 'M' (Integer Multiplication and Division)}}
