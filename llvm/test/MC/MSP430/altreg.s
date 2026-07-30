; RUN: llvm-mc -triple msp430 -show-encoding < %s | FileCheck %s
  mov pc, r0 ; CHECK: mov r0, r0
  mov sp, r1 ; CHECK: mov r1, r1
  mov sr, r2 ; CHECK: mov r2, r2
  mov cg, r3 ; CHECK: mov r3, r3
  mov fp, r4 ; CHECK: mov r4, r4
  bic #8, SR ; CHECK: dint
