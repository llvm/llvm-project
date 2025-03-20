; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -dr - | FileCheck --check-prefix=INST %s

foo:
  brbc 3, .+8
  brbc 0, .-16
  .short 0xf759
  .short 0xf752
  .short 0xf74c
  .short 0xf4c7

; CHECK: brvc (.Ltmp0+8)+2   ; encoding: [0bAAAAA011,0b111101AA]
; CHECK-NEXT:                ; fixup A - offset: 0, value: (.Ltmp0+8)+2, kind: fixup_7_pcrel
;
; CHECK: brcc (.Ltmp1-16)+2  ; encoding: [0bAAAAA000,0b111101AA]
; CHECK-NEXT:                ; fixup A - offset: 0, value: (.Ltmp1-16)+2, kind: fixup_7_pcrel

; INST-LABEL: <foo>:
; INST-NEXT: 23 f4   brvc .+8
; INST-NEXT: c0 f7   brsh .-16
; INST-NEXT: 59 f7   brne .-42
; INST-NEXT: 52 f7   brpl .-44
; INST-NEXT: 4c f7   brge .-46
; INST-NEXT: c7 f4   brid .+48
