; RUN: llvm-mc -triple avr -mattr=mul -show-encoding < %s | FileCheck %s


foo:

  muls r22, r16
  muls r19, r17
  muls r28, r31
  muls r31, r31

; CHECK: muls r22, r16                   ; encoding: [0x60,0x02]
; CHECK: muls r19, r17                   ; encoding: [0x31,0x02]
; CHECK: muls r28, r31                   ; encoding: [0xcf,0x02]
; CHECK: muls r31, r31                   ; encoding: [0xff,0x02]
