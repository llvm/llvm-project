; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s | llvm-objdump -dr - | FileCheck -check-prefix=CHECK-INST %s

foo:
  lsr r31
  lsr r25
  lsr r5
  lsr r0

; CHECK: lsr r31                ; encoding: [0xf6,0x95]
; CHECK: lsr r25                ; encoding: [0x96,0x95]
; CHECK: lsr r5                 ; encoding: [0x56,0x94]
; CHECK: lsr r0                 ; encoding: [0x06,0x94]

; CHECK-INST: lsr r31
; CHECK-INST: lsr r25
; CHECK-INST: lsr r5
; CHECK-INST: lsr r0
