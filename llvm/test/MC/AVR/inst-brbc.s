; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s


foo:

  brbc 3, .+8
  brbc 0, .-16

; CHECK: brvc .Ltmp0+8              ; encoding: [0bAAAAA011,0b111101AA]
; CHECK:                            ; fixup A - offset: 0, value: .Ltmp0+8, kind: fixup_7_pcrel
; CHECK: brcc .Ltmp1-16             ; encoding: [0bAAAAA000,0b111101AA]
; CHECK:                            ; fixup A - offset: 0, value: .Ltmp1-16, kind: fixup_7_pcrel
