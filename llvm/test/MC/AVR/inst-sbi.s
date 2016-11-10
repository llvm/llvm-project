; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s


foo:

  sbi 3, 5
  sbi 1, 1
  sbi 0, 0
  sbi 7, 2

  sbi main, 0

; CHECK: sbi 3, 5                  ; encoding: [0x1d,0x9a]
; CHECK: sbi 1, 1                  ; encoding: [0x09,0x9a]
; CHECK: sbi 0, 0                  ; encoding: [0x00,0x9a]
; CHECK: sbi 7, 2                  ; encoding: [0x3a,0x9a]

; CHECK: sbi main, 0               ; encoding: [0bAAAAA000,0x9a]
; CHECK:                           ;   fixup A - offset: 0, value: main, kind: fixup_port5

