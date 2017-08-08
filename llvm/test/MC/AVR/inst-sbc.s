; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s


foo:

  sbc r0,  r15
  sbc r15, r0
  sbc r16, r31
  sbc r31, r16

; CHECK: sbc r0,  r15               ; encoding: [0x0f,0x08]
; CHECK: sbc r15, r0                ; encoding: [0xf0,0x08]
; CHECK: sbc r16, r31               ; encoding: [0x0f,0x0b]
; CHECK: sbc r31, r16               ; encoding: [0xf0,0x0b]
