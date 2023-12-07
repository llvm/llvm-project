; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -d - | FileCheck --check-prefix=INST %s

foo:

  brbc 3, .+8
  brbc 0, .-16
  .short 0xf759
  .short 0xf752
  .short 0xf74c
  .short 0xf4c7

; CHECK: brvc .Ltmp0+8              ; encoding: [0bAAAAA011,0b111101AA]
; CHECK:                            ; fixup A - offset: 0, value: .Ltmp0+8, kind: fixup_7_pcrel
; CHECK: brcc .Ltmp1-16             ; encoding: [0bAAAAA000,0b111101AA]
; CHECK:                            ; fixup A - offset: 0, value: .Ltmp1-16, kind: fixup_7_pcrel

; INST: 23 f4   brvc .+8
; INST: c0 f7   brsh .-16
; INST: 59 f7   brne .-42
; INST: 52 f7   brpl .-44
; INST: 4c f7   brge .-46
; INST: c7 f4   brid .+48
