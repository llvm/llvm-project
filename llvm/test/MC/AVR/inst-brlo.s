; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
;
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -dr - \
; RUN:     | FileCheck --check-prefix=INST %s

foo:
  brlo .+12
  brlo .+28
  brlo bar

bar:

; CHECK: brlo .Ltmp0+12+2  ; encoding: [0bAAAAA000,0b111100AA]
; CHECK: brlo .Ltmp1+28+2  ; encoding: [0bAAAAA000,0b111100AA]
; CHECK: brlo bar            ; encoding: [0bAAAAA000,0b111100AA]

; INST-LABEL: <foo>:
; INST-NEXT: f8 f3      brlo .-2
; INST-NEXT: R_AVR_7_PCREL .text+0xe
; INST-NEXT: f8 f3      brlo .-2
; INST-NEXT: R_AVR_7_PCREL .text+0x20
; INST-NEXT: f8 f3      brlo .-2
; INST-NEXT: R_AVR_7_PCREL .text+0x6
