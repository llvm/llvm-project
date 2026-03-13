; RUN: llvm-mc -triple avr -mattr=mul -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr -mattr=mul < %s | llvm-objdump -dr --mattr=mul - | FileCheck -check-prefix=CHECK-INST %s

foo:
  fmulsu r22, r16
  fmulsu r19, r17
  fmulsu r21, r23
  fmulsu r23, r23

; CHECK: fmulsu r22, r16                   ; encoding: [0xe8,0x03]
; CHECK: fmulsu r19, r17                   ; encoding: [0xb9,0x03]
; CHECK: fmulsu r21, r23                   ; encoding: [0xdf,0x03]
; CHECK: fmulsu r23, r23                   ; encoding: [0xff,0x03]

; CHECK-INST: fmulsu r22, r16
; CHECK-INST: fmulsu r19, r17
; CHECK-INST: fmulsu r21, r23
; CHECK-INST: fmulsu r23, r23
