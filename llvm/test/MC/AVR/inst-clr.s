; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s


foo:

  clr r2
  clr r12
  clr r5
  clr r0

; CHECK: eor r2,  r2                  ; encoding: [0x22,0x24]
; CHECK: eor r12, r12                 ; encoding: [0xcc,0x24]
; CHECK: eor r5,  r5                  ; encoding: [0x55,0x24]
; CHECK: eor r0,  r0                  ; encoding: [0x00,0x24]
