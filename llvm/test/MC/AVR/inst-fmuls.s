; RUN: llvm-mc -triple avr -mattr=mul -show-encoding < %s | FileCheck %s


foo:

  fmuls r22, r16
  fmuls r19, r17
  fmuls r21, r23
  fmuls r23, r23

; CHECK: fmuls r22, r16                   ; encoding: [0xe0,0x03]
; CHECK: fmuls r19, r17                   ; encoding: [0xb1,0x03]
; CHECK: fmuls r21, r23                   ; encoding: [0xd7,0x03]
; CHECK: fmuls r23, r23                   ; encoding: [0xf7,0x03]
