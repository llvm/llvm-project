; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
;
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -d - \
; RUN:     | FileCheck --check-prefix=INST %s

foo:
  rcall  .+0
  rcall  .-8
  rcall  .+12
  rcall  .+46
  .short  0xdfea

; CHECK: rcall (.Ltmp0+0)+2   ; encoding: [A,0b1101AAAA]
; CHECK-NEXT:                 ;   fixup A - offset: 0, value: (.Ltmp0+0)+2, kind: fixup_13_pcrel
; CHECK: rcall (.Ltmp1-8)+2   ; encoding: [A,0b1101AAAA]
; CHECK-NEXT:                 ;   fixup A - offset: 0, value: (.Ltmp1-8)+2, kind: fixup_13_pcrel
; CHECK: rcall (.Ltmp2+12)+2  ; encoding: [A,0b1101AAAA]
; CHECK-NEXT:                 ;   fixup A - offset: 0, value: (.Ltmp2+12)+2, kind: fixup_13_pcrel
; CHECK: rcall (.Ltmp3+46)+2  ; encoding: [A,0b1101AAAA]
; CHECK-NEXT:                 ;   fixup A - offset: 0, value: (.Ltmp3+46)+2, kind: fixup_13_pcrel

; INST-LABEL: <foo>:
; INST-NEXT: 00 d0    rcall .+0
; INST-NEXT: fc df    rcall .-8
; INST-NEXT: 06 d0    rcall .+12
; INST-NEXT: 17 d0    rcall .+46
; INST-NEXT: ea df    rcall .-44
