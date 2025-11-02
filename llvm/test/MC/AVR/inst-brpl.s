; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
;
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -dr - \
; RUN:     | FileCheck --check-prefix=INST %s

foo:
  brpl .-12
  brpl .+18
  brpl bar

bar:

; CHECK: brpl .Ltmp0-12+2  ; encoding: [0bAAAAA010,0b111101AA]
; CHECK: brpl .Ltmp1+18+2  ; encoding: [0bAAAAA010,0b111101AA]
; CHECK: brpl bar            ; encoding: [0bAAAAA010,0b111101AA]

; INST-LABEL: <foo>:
; INST-NEXT: fa f7      brpl .-2
; INST-NEXT: R_AVR_7_PCREL .text-0xa
; INST-NEXT: fa f7      brpl .-2
; INST-NEXT: R_AVR_7_PCREL .text+0x16
; INST-NEXT: fa f7      brpl .-2
; INST-NEXT: R_AVR_7_PCREL .text+0x6
