; RUN: llvm-mc -triple msp430 -show-encoding < %s | FileCheck %s
  mov #4, r15 ; CHECK: mov #4, r15 ; encoding: [0x2f,0x42]
  mov #8, r15 ; CHECK: mov #8, r15 ; encoding: [0x3f,0x42]
  mov #0, r15 ; CHECK: clr r15     ; encoding: [0x0f,0x43]
  mov #1, r15 ; CHECK: mov #1, r15 ; encoding: [0x1f,0x43]
  mov #2, r15 ; CHECK: mov #2, r15 ; encoding: [0x2f,0x43]
  mov #-1, r7 ; CHECK: mov #-1, r7 ; encoding: [0x37,0x43]

  push #8     ; CHECK: push #8     ; encoding: [0x32,0x12]
  push #42    ; CHECK: push #42    ; encoding: [0x30,0x12,0x2a,0x00]
