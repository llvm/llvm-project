; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s


foo:

  bld r3, 5
  bld r1, 1
  bld r0, 0
  bld r7, 2

; CHECK: bld r3, 5                  ; encoding: [0x35,0xf8]
; CHECK: bld r1, 1                  ; encoding: [0x11,0xf8]
; CHECK: bld r0, 0                  ; encoding: [0x00,0xf8]
; CHECK: bld r7, 2                  ; encoding: [0x72,0xf8]
