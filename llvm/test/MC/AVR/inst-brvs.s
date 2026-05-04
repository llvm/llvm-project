; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
;
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -dr - \
; RUN:     | FileCheck --check-prefix=INST %s

foo:
  brvs .+18
  brvs .+32
  brvs bar

bar:

; CHECK: brvs .Ltmp0+18+2  ; encoding: [0bAAAAA011,0b111100AA]
; CHECK: brvs .Ltmp1+32+2  ; encoding: [0bAAAAA011,0b111100AA]
; CHECK: brvs bar            ; encoding: [0bAAAAA011,0b111100AA]

; INST-LABEL: <foo>:
; INST-NEXT: fb f3      brvs .-2
; INST-NEXT: R_AVR_7_PCREL .text+0x14
; INST-NEXT: fb f3      brvs .-2
; INST-NEXT: R_AVR_7_PCREL .text+0x24
; INST-NEXT: fb f3      brvs .-2
; INST-NEXT: R_AVR_7_PCREL .text+0x6
