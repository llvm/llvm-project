; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s


foo:

  and r0,  r15
  and r15, r0
  and r16, r31
  and r31, r16

; CHECK: and r0,  r15               ; encoding: [0x0f,0x20]
; CHECK: and r15, r0                ; encoding: [0xf0,0x20]
; CHECK: and r16, r31               ; encoding: [0x0f,0x23]
; CHECK: and r31, r16               ; encoding: [0xf0,0x23]
