; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s


foo:

  bst r3, 5
  bst r1, 1
  bst r0, 0
  bst r7, 2

; CHECK: bst r3, 5                  ; encoding: [0x35,0xfa]
; CHECK: bst r1, 1                  ; encoding: [0x11,0xfa]
; CHECK: bst r0, 0                  ; encoding: [0x00,0xfa]
; CHECK: bst r7, 2                  ; encoding: [0x72,0xfa]
