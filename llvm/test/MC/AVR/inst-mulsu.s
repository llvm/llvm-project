; RUN: llvm-mc -triple avr -mattr=mul -show-encoding < %s | FileCheck %s


foo:

  mulsu r22, r16
  mulsu r19, r17
  mulsu r21, r23
  mulsu r23, r23

; CHECK: mulsu r22, r16                   ; encoding: [0x60,0x03]
; CHECK: mulsu r19, r17                   ; encoding: [0x31,0x03]
; CHECK: mulsu r21, r23                   ; encoding: [0x57,0x03]
; CHECK: mulsu r23, r23                   ; encoding: [0x77,0x03]
