; RUN: llvm-mc -triple avr -mattr=sram -show-encoding < %s | FileCheck %s


foo:

  push r31
  push r25
  push r5
  push r0

; CHECK: push r31                ; encoding: [0xff,0x93]
; CHECK: push r25                ; encoding: [0x9f,0x93]
; CHECK: push r5                 ; encoding: [0x5f,0x92]
; CHECK: push r0                 ; encoding: [0x0f,0x92]
