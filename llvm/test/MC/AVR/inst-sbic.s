; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s


foo:

  sbic 4,  3
  sbic 6,  2
  sbic 16, 5
  sbic 0,  0

  sbic foo+1, 1

; CHECK: sbic 4,  3                  ; encoding: [0x23,0x99]
; CHECK: sbic 6,  2                  ; encoding: [0x32,0x99]
; CHECK: sbic 16, 5                  ; encoding: [0x85,0x99]
; CHECK: sbic 0,  0                  ; encoding: [0x00,0x99]

; CHECK: sbic foo+1, 1               ; encoding: [0bAAAAA001,0x99]
; CHECK:                             ;   fixup A - offset: 0, value: foo+1, kind: fixup_port5
