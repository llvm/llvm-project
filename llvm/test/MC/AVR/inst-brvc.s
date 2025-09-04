; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
;
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -dr - \
; RUN:     | FileCheck --check-prefix=INST %s

foo:
  brvc .-28
  brvc .-62
  brvc bar

bar:

; CHECK: brvc .Ltmp0-28+2  ; encoding: [0bAAAAA011,0b111101AA]
; CHECK: brvc .Ltmp1-62+2  ; encoding: [0bAAAAA011,0b111101AA]
; CHECK: brvc bar            ; encoding: [0bAAAAA011,0b111101AA]

; INST-LABEL: <foo>:
; INST-NEXT: fb f7      brvc .-2
; INST-NEXT: R_AVR_7_PCREL .text-0x1a
; INST-NEXT: fb f7      brvc .-2
; INST-NEXT: R_AVR_7_PCREL .text-0x3a
; INST-NEXT: fb f7      brvc .-2
; INST-NEXT: R_AVR_7_PCREL .text+0x6
