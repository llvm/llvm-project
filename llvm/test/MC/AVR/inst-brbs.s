; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -dr - | FileCheck --check-prefix=INST %s

foo:
  brbs 3, .+8
  brbs 0, .-12
  .short 0xf359
  .short 0xf352
  .short 0xf34c
  .short 0xf077

; CHECK: brvs .Ltmp0+8+2   ; encoding: [0bAAAAA011,0b111100AA]
; CHECK: brcs .Ltmp1-12+2  ; encoding: [0bAAAAA000,0b111100AA]

; INST-LABEL: <foo>:
; INST-NEXT: fb f3   brvs .-2
; INST-NEXT: R_AVR_7_PCREL .text+0xa
; INST-NEXT: f8 f3   brlo .-2
; INST-NEXT: R_AVR_7_PCREL .text-0x8
; INST-NEXT: 59 f3   breq .-42
; INST-NEXT: 52 f3   brmi .-44
; INST-NEXT: 4c f3   brlt .-46
; INST-NEXT: 77 f0   brie .+28
