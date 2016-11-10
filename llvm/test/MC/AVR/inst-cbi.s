; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s


foo:

  cbi 3, 5
  cbi 1, 1
  cbi 0, 0
  cbi 7, 2

  cbi bar-2, 2

; CHECK: cbi 3, 5                  ; encoding: [0x1d,0x98]
; CHECK: cbi 1, 1                  ; encoding: [0x09,0x98]
; CHECK: cbi 0, 0                  ; encoding: [0x00,0x98]
; CHECK: cbi 7, 2                  ; encoding: [0x3a,0x98]

; CHECK: cbi bar-2, 2              ; encoding: [0bAAAAA010,0x98]
; CHECK:                           ;   fixup A - offset: 0, value: bar-2, kind: fixup_port5

