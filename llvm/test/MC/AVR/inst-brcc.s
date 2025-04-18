; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
;
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -dr - \
; RUN:     | FileCheck --check-prefix=INST %s

foo:
  brcc .+66
  brcc .-22
  brbc 0, .+66
  brbc 0, bar

bar:

; CHECK: brcc .Ltmp0+66+2  ; encoding: [0bAAAAA000,0b111101AA]
; CHECK: brcc .Ltmp1-22+2  ; encoding: [0bAAAAA000,0b111101AA]
; CHECK: brcc .Ltmp2+66+2  ; encoding: [0bAAAAA000,0b111101AA]
; CHECK: brcc bar            ; encoding: [0bAAAAA000,0b111101AA]

; INST-LABEL: <foo>:
; INST-NEXT: 08 f5      brsh .+66
; INST-NEXT: a8 f7      brsh .-22
; INST-NEXT: 08 f5      brsh .+66
; INST-NEXT: 00 f4      brsh .+0
