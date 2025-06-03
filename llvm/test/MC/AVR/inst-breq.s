; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
;
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -dr - \
; RUN:     | FileCheck --check-prefix=INST %s

foo:
  breq .-18
  breq .-12
  brbs 1, .-18
  brbs 1, bar

bar:

; CHECK: breq    .Ltmp0-18+2     ; encoding: [0bAAAAA001,0b111100AA]
; CHECK: breq    .Ltmp1-12+2     ; encoding: [0bAAAAA001,0b111100AA]
; CHECK: brbs    1, .Ltmp2-18+2  ; encoding: [0bAAAAA001,0b111100AA]
; CHECK: brbs    1, bar            ; encoding: [0bAAAAA001,0b111100AA]

; INST-LABEL: <foo>:
; INST-NEXT: b9 f3      breq .-18
; INST-NEXT: d1 f3      breq .-12
; INST-NEXT: b9 f3      breq .-18
; INST-NEXT: 01 f0      breq .+0
