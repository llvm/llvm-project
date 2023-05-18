; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -d - | FileCheck --check-prefix=INST %s

foo:

  brbs 3, .+8
  brbs 0, .-12
  .short 0xf359
  .short 0xf352
  .short 0xf34c
  .short 0xf077

; CHECK: brvs .Ltmp0+8              ; encoding: [0bAAAAA011,0b111100AA]
; CHECK:                            ; fixup A - offset: 0, value: .Ltmp0+8, kind: fixup_7_pcrel
; CHECK: brcs .Ltmp1-12             ; encoding: [0bAAAAA000,0b111100AA]
; CHECK:                            ; fixup A - offset: 0, value: .Ltmp1-12, kind: fixup_7_pcrel

; INST: 23 f0   brvs .+8
; INST: d0 f3   brlo .-12
; INST: 59 f3   breq .-42
; INST: 52 f3   brmi .-44
; INST: 4c f3   brlt .-46
; INST: 77 f0   brie .+28
