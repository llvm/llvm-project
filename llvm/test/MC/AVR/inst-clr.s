; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s


foo:

  clr r2
  clr r12
  clr r5
  clr r0

; CHECK: clr r2                  ; encoding: [0x22,0x24]
; CHECK: clr r12                 ; encoding: [0xcc,0x24]
; CHECK: clr r5                  ; encoding: [0x55,0x24]
; CHECK: clr r0                  ; encoding: [0x00,0x24]

