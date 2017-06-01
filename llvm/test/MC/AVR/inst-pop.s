; RUN: llvm-mc -triple avr -mattr=sram -show-encoding < %s | FileCheck %s


foo:

  pop r31
  pop r25
  pop r5
  pop r0

; CHECK: pop r31                ; encoding: [0xff,0x91]
; CHECK: pop r25                ; encoding: [0x9f,0x91]
; CHECK: pop r5                 ; encoding: [0x5f,0x90]
; CHECK: pop r0                 ; encoding: [0x0f,0x90]
