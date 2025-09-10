; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
;
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -dr - \
; RUN:     | FileCheck --check-prefix=INST %s

foo:
  rcall  .+0
  rcall  .-8
  rcall  .+12
  rcall  .+46
  .short  0xdfea

; CHECK: rcall .Ltmp0+0+2   ; encoding: [A,0b1101AAAA]
; CHECK: rcall .Ltmp1-8+2   ; encoding: [A,0b1101AAAA]
; CHECK: rcall .Ltmp2+12+2  ; encoding: [A,0b1101AAAA]
; CHECK: rcall .Ltmp3+46+2  ; encoding: [A,0b1101AAAA]

; INST-LABEL: <foo>:
; INST-NEXT: ff df    rcall .-2
; INST-NEXT: R_AVR_13_PCREL .text+0x2
; INST-NEXT: ff df    rcall .-2
; INST-NEXT: R_AVR_13_PCREL .text-0x4
; INST-NEXT: ff df    rcall .-2
; INST-NEXT: R_AVR_13_PCREL .text+0x12
; INST-NEXT: ff df    rcall .-2
; INST-NEXT: R_AVR_13_PCREL .text+0x36
