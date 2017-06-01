; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s


foo:

  cbr r17, 208
  cbr r24, 190
  cbr r20, 173
  cbr r31, 0

; CHECK: cbr r17, 208                 ; encoding: [0x1f,0x72]
; CHECK: cbr r24, 190                 ; encoding: [0x81,0x74]
; CHECK: cbr r20, 173                 ; encoding: [0x42,0x75]
; CHECK: cbr r31, 0                   ; encoding: [0xff,0x7f]
