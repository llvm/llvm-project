; RUN: llvm-mc -triple avr -mattr=mul -show-encoding < %s | FileCheck %s


foo:

  fmul r22, r16
  fmul r19, r17
  fmul r21, r23
  fmul r23, r23

; CHECK: fmul r22, r16                   ; encoding: [0x68,0x03]
; CHECK: fmul r19, r17                   ; encoding: [0x39,0x03]
; CHECK: fmul r21, r23                   ; encoding: [0x5f,0x03]
; CHECK: fmul r23, r23                   ; encoding: [0x7f,0x03]
