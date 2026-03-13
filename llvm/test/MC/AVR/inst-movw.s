; RUN: llvm-mc -triple avr -mattr=movw -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr -mattr=movw < %s | llvm-objdump -dr --mattr=movw - | FileCheck -check-prefix=CHECK-INST %s

foo:
  movw r10, r8
  movw r12, r16
  movw r20, r22
  movw r8,  r12
  movw r0,  r0
  movw r0,  r30
  movw r30, r30
  movw r30, r0

; CHECK: movw r10, r8            ; encoding: [0x54,0x01]
; CHECK: movw r12, r16           ; encoding: [0x68,0x01]
; CHECK: movw r20, r22           ; encoding: [0xab,0x01]
; CHECK: movw r8,  r12           ; encoding: [0x46,0x01]
; CHECK: movw r0,  r0            ; encoding: [0x00,0x01]
; CHECK: movw r0,  r30           ; encoding: [0x0f,0x01]
; CHECK: movw r30, r30           ; encoding: [0xff,0x01]
; CHECK: movw r30, r0            ; encoding: [0xf0,0x01]

; CHECK-INST: movw r10, r8
; CHECK-INST: movw r12, r16
; CHECK-INST: movw r20, r22
; CHECK-INST: movw r8,  r12
; CHECK-INST: movw r0,  r0
; CHECK-INST: movw r0,  r30
; CHECK-INST: movw r30, r30
; CHECK-INST: movw r30, r0
