; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
;
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -dr - \
; RUN:     | FileCheck --check-prefix=INST %s

foo:
  brhc .+12
  brhc .+14
  brhc bar

bar:

; CHECK: brhc .Ltmp0+12+2  ; encoding: [0bAAAAA101,0b111101AA]
; CHECK: brhc .Ltmp1+14+2  ; encoding: [0bAAAAA101,0b111101AA]
; CHECK: brhc bar            ; encoding: [0bAAAAA101,0b111101AA]

; INST-LABEL: <foo>:
; INST-NEXT: fd f7      brhc .-2
; INST-NEXT: R_AVR_7_PCREL .text+0xe
; INST-NEXT: fd f7      brhc .-2
; INST-NEXT: R_AVR_7_PCREL .text+0x12
; INST-NEXT: fd f7      brhc .-2
; INST-NEXT: R_AVR_7_PCREL .text+0x6
