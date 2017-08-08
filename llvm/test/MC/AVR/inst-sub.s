; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s


foo:
  sub r0,  r15
  sub r15, r0
  sub r16, r31
  sub r31, r16

; CHECK: sub r0,  r15               ; encoding: [0x0f,0x18]
; CHECK: sub r15, r0                ; encoding: [0xf0,0x18]
; CHECK: sub r16, r31               ; encoding: [0x0f,0x1b]
; CHECK: sub r31, r16               ; encoding: [0xf0,0x1b]
