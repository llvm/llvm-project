; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
;
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -dr - \
; RUN:     | FileCheck --check-prefix=INST %s

foo:
  brhs .-66
  brhs .+14
  brhs bar

bar:

; CHECK: brhs .Ltmp0-66+2  ; encoding: [0bAAAAA101,0b111100AA]
; CHECK: brhs .Ltmp1+14+2  ; encoding: [0bAAAAA101,0b111100AA]
; CHECK: brhs bar            ; encoding: [0bAAAAA101,0b111100AA]

; INST-LABEL: <foo>:
; INST-NEXT: fd f3      brhs .-2
; INST-NEXT: R_AVR_7_PCREL .text-0x40
; INST-NEXT: fd f3      brhs .-2
; INST-NEXT: R_AVR_7_PCREL .text+0x12
; INST-NEXT: fd f3      brhs .-2
; INST-NEXT: R_AVR_7_PCREL .text+0x6
