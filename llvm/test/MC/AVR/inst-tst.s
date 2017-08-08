; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s


foo:

  tst r3
  tst r14
  tst r24
  tst r12

; CHECK: tst r3               ; encoding: [0x33,0x20]
; CHECK: tst r14              ; encoding: [0xee,0x20]
; CHECK: tst r24              ; encoding: [0x88,0x23]
; CHECK: tst r12              ; encoding: [0xcc,0x20]
