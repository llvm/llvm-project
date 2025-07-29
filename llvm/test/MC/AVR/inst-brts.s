; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
;
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -dr - \
; RUN:     | FileCheck --check-prefix=INST %s

foo:
  brts .+18
  brts .+22
  brts bar

bar:

; CHECK: brts .Ltmp0+18+2  ; encoding: [0bAAAAA110,0b111100AA]
; CHECK: brts .Ltmp1+22+2  ; encoding: [0bAAAAA110,0b111100AA]
; CHECK: brts bar            ; encoding: [0bAAAAA110,0b111100AA]

; INST-LABEL: <foo>:
; INST-NEXT: fe f3      brts .-2
; INST-NEXT: R_AVR_7_PCREL .text+0x14
; INST-NEXT: fe f3      brts .-2
; INST-NEXT: R_AVR_7_PCREL .text+0x1a
; INST-NEXT: fe f3      brts .-2
; INST-NEXT: R_AVR_7_PCREL .text+0x6
