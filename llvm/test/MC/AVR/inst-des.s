; RUN: llvm-mc -triple avr -mattr=des -show-encoding < %s | FileCheck %s


foo:

  des 0
  des 6
  des 1
  des 8

; CHECK: des 0                  ; encoding: [0x0b,0x94]
; CHECK: des 6                  ; encoding: [0x6b,0x94]
; CHECK: des 1                  ; encoding: [0x1b,0x94]
; CHECK: des 8                  ; encoding: [0x8b,0x94]
