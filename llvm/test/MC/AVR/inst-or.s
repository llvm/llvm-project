; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s


foo:
  or r0,  r15
  or r15, r0
  or r16, r31
  or r31, r16

; CHECK: or r0,  r15               ; encoding: [0x0f,0x28]
; CHECK: or r15, r0                ; encoding: [0xf0,0x28]
; CHECK: or r16, r31               ; encoding: [0x0f,0x2b]
; CHECK: or r31, r16               ; encoding: [0xf0,0x2b]
