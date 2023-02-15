; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -d - | FileCheck --check-prefix=INST %s


foo:

  rcall  .+0
  rcall  .-8
  rcall  .+12
  rcall  .+46
  .short  0xdfea

; CHECK: rcall  .Ltmp0+0             ; encoding: [A,0b1101AAAA]
; CHECK:                             ;   fixup A - offset: 0, value: .Ltmp0+0, kind: fixup_13_pcrel
; CHECK: rcall  .Ltmp1-8             ; encoding: [A,0b1101AAAA]
; CHECK:                             ;   fixup A - offset: 0, value: .Ltmp1-8, kind: fixup_13_pcrel
; CHECK: rcall  .Ltmp2+12            ; encoding: [A,0b1101AAAA]
; CHECK:                             ;   fixup A - offset: 0, value: .Ltmp2+12, kind: fixup_13_pcrel
; CHECK: rcall  .Ltmp3+46            ; encoding: [A,0b1101AAAA]
; CHECK:                             ;   fixup A - offset: 0, value: .Ltmp3+46, kind: fixup_13_pcrel

; INST: 00 d0    rcall .+0
; INST: fc df    rcall .-8
; INST: 06 d0    rcall .+12
; INST: 17 d0    rcall .+46
; INST: ea df    rcall .-44
