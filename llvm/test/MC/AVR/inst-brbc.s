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

; INST: brvc .+0
; INST: brsh .+0
; INST: brne .-42
; INST: brpl .-44
; INST: brge .-46
; INST: brid .+48
