; RUN: llvm-mc -triple avr -mattr=mul -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr -mattr=mul < %s | llvm-objdump -d --mattr=mul - | FileCheck -check-prefix=CHECK-INST %s


foo:

  mulsu r22, r16
  mulsu r19, r17
  mulsu r21, r23
  mulsu r23, r23

; CHECK: mulsu r22, r16                   ; encoding: [0x60,0x03]
; CHECK: mulsu r19, r17                   ; encoding: [0x31,0x03]
; CHECK: mulsu r21, r23                   ; encoding: [0x57,0x03]
; CHECK: mulsu r23, r23                   ; encoding: [0x77,0x03]

; CHECK-INST: mulsu r22, r16
; CHECK-INST: mulsu r19, r17
; CHECK-INST: mulsu r21, r23
; CHECK-INST: mulsu r23, r23
