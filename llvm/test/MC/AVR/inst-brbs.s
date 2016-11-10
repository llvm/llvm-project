; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s


foo:

  brbs 3, .+8
  brbs 0, .-12

; CHECK: brvs .Ltmp0+8              ; encoding: [0bAAAAA011,0b111100AA]
; CHECK:                            ; fixup A - offset: 0, value: .Ltmp0+8, kind: fixup_7_pcrel
; CHECK: brcs .Ltmp1-12             ; encoding: [0bAAAAA000,0b111100AA]
; CHECK:                            ; fixup A - offset: 0, value: .Ltmp1-12, kind: fixup_7_pcrel
