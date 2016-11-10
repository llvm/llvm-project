; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s


foo:
  sbis 4,  3
  sbis 6,  2
  sbis 16, 5
  sbis 0,  0

  sbis FOO+4, 7

; CHECK: sbis 4,  3                  ; encoding: [0x23,0x9b]
; CHECK: sbis 6,  2                  ; encoding: [0x32,0x9b]
; CHECK: sbis 16, 5                  ; encoding: [0x85,0x9b]
; CHECK: sbis 0,  0                  ; encoding: [0x00,0x9b]

; CHECK: sbis FOO+4, 7               ; encoding: [0bAAAAA111,0x9b]
; CHECK:                             ;   fixup A - offset: 0, value: FOO+4, kind: fixup_port5
