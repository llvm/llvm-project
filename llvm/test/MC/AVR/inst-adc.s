; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s


foo:

  adc r0,  r15
  adc r15, r0
  adc r16, r31
  adc r31, r16

; CHECK: adc r0,  r15               ; encoding: [0x0f,0x1c]
; CHECK: adc r15, r0                ; encoding: [0xf0,0x1c]
; CHECK: adc r16, r31               ; encoding: [0x0f,0x1f]
; CHECK: adc r31, r16               ; encoding: [0xf0,0x1f]
