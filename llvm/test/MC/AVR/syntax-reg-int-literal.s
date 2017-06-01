; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s

foo:
  add 0,   15
  add 15,  r0
  add 16,  31
  add r16, r31
  add 31,  r16


; CHECK: add r0,  r15               ; encoding: [0x0f,0x0c]
; CHECK: add r15, r0                ; encoding: [0xf0,0x0c]
; CHECK: add r16, r31               ; encoding: [0x0f,0x0f]
; CHECK: add r16, r31               ; encoding: [0x0f,0x0f]
; CHECK: add r31, r16               ; encoding: [0xf0,0x0f]
