; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s | llvm-objdump -dr - | FileCheck -check-prefix=CHECK-INST %s

foo:
  swap r31
  swap r25
  swap r5
  swap r0

; CHECK: swap r31                ; encoding: [0xf2,0x95]
; CHECK: swap r25                ; encoding: [0x92,0x95]
; CHECK: swap r5                 ; encoding: [0x52,0x94]
; CHECK: swap r0                 ; encoding: [0x02,0x94]

; CHECK-INST: swap r31
; CHECK-INST: swap r25
; CHECK-INST: swap r5
; CHECK-INST: swap r0
