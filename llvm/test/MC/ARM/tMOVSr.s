@ REQUIRES: asserts
@ RUN: llvm-mc --triple=thumbv8 --debug %s 2>&1 | FileCheck %s --match-full-lines

@ CHECK: Changed to: <MCInst #{{[0-9]+}} tMOVSr <MCOperand Reg:{{[0-9]+}}> <MCOperand Reg:{{[0-9]+}}>>
.text
  movs r2, r3
