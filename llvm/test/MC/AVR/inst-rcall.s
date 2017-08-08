; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s


foo:

  rcall  .+0
  rcall  .-8
  rcall  .+12
  rcall  .+46

; CHECK: rcall  .Ltmp0+0             ; encoding: [A,0b1101AAAA]
; CHECK:                             ;   fixup A - offset: 0, value: .Ltmp0+0, kind: fixup_13_pcrel
; CHECK: rcall  .Ltmp1-8             ; encoding: [A,0b1101AAAA]
; CHECK:                             ;   fixup A - offset: 0, value: .Ltmp1-8, kind: fixup_13_pcrel
; CHECK: rcall  .Ltmp2+12            ; encoding: [A,0b1101AAAA]
; CHECK:                             ;   fixup A - offset: 0, value: .Ltmp2+12, kind: fixup_13_pcrel
; CHECK: rcall  .Ltmp3+46            ; encoding: [A,0b1101AAAA]
; CHECK:                             ;   fixup A - offset: 0, value: .Ltmp3+46, kind: fixup_13_pcrel

