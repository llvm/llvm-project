; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s


foo:

  eor r0,  r15
  eor r15, r0
  eor r16, r31
  eor r31, r16

; CHECK: eor r0,  r15               ; encoding: [0x0f,0x24]
; CHECK: eor r15, r0                ; encoding: [0xf0,0x24]
; CHECK: eor r16, r31               ; encoding: [0x0f,0x27]
; CHECK: eor r31, r16               ; encoding: [0xf0,0x27]
