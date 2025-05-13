; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
;
; RUN: llvm-mc -filetype=obj -triple avr < %s \
; RUN:     | llvm-objdump -dr - \
; RUN:     | FileCheck --check-prefix=INST %s

foo:
  brsh .+32
  brsh .+70
  brsh bar

bar:

; CHECK: brsh .Ltmp0+32+2  ; encoding: [0bAAAAA000,0b111101AA]
; CHECK: brsh .Ltmp1+70+2  ; encoding: [0bAAAAA000,0b111101AA]
; CHECK: brsh bar            ; encoding: [0bAAAAA000,0b111101AA]

; INST-LABEL: <foo>:
; INST-NEXT: 80 f4      brsh .+32
; INST-NEXT: 18 f5      brsh .+70
; INST-NEXT: 00 f4      brsh .+0
