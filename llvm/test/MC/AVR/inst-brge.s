; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
;
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -dr - \
; RUN:     | FileCheck --check-prefix=INST %s

foo:
  brge .+50
  brge .+42
  brge bar

bar:

; CHECK: brge .Ltmp0+50+2  ; encoding: [0bAAAAA100,0b111101AA]
; CHECK: brge .Ltmp1+42+2  ; encoding: [0bAAAAA100,0b111101AA]
; CHECK: brge bar            ; encoding: [0bAAAAA100,0b111101AA]

; INST-LABEL: <foo>:
; INST-NEXT: fc f7      brge .-2
; INST-NEXT: R_AVR_7_PCREL .text+0x34
; INST-NEXT: fc f7      brge .-2
; INST-NEXT: R_AVR_7_PCREL .text+0x2e
; INST-NEXT: fc f7      brge .-2
; INST-NEXT: R_AVR_7_PCREL .text+0x6
