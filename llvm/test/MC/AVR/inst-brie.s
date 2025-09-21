; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
;
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -dr - \
; RUN:     | FileCheck --check-prefix=INST %s

foo:
  brie .+20
  brie .+40
  brie bar

bar:

; CHECK: brie .Ltmp0+20+2  ; encoding: [0bAAAAA111,0b111100AA]
; CHECK: brie .Ltmp1+40+2  ; encoding: [0bAAAAA111,0b111100AA]
; CHECK: brie bar            ; encoding: [0bAAAAA111,0b111100AA]

; INST-LABEL: <foo>:
; INST-NEXT: ff f3      brie .-2
; INST-NEXT: R_AVR_7_PCREL .text+0x16
; INST-NEXT: ff f3      brie .-2
; INST-NEXT: R_AVR_7_PCREL .text+0x2c
; INST-NEXT: ff f3      brie .-2
; INST-NEXT: R_AVR_7_PCREL .text+0x6
