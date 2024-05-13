; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s | llvm-objdump --no-print-imm-hex -d - | FileCheck -check-prefix=CHECK-INST %s


foo:
  sbis 4,  3
  sbis 6,  2
  sbis 16, 5
  sbis 0,  0
  sbis 31, 0
  sbis 0,  7
  sbis 31, 7

  sbis FOO+4, 7

; CHECK: sbis 4,  3                  ; encoding: [0x23,0x9b]
; CHECK: sbis 6,  2                  ; encoding: [0x32,0x9b]
; CHECK: sbis 16, 5                  ; encoding: [0x85,0x9b]
; CHECK: sbis 0,  0                  ; encoding: [0x00,0x9b]
; CHECK: sbis 31, 0                  ; encoding: [0xf8,0x9b]
; CHECK: sbis 0,  7                  ; encoding: [0x07,0x9b]
; CHECK: sbis 31, 7                  ; encoding: [0xff,0x9b]

; CHECK: sbis FOO+4, 7               ; encoding: [0bAAAAA111,0x9b]
; CHECK:                             ;   fixup A - offset: 0, value: FOO+4, kind: fixup_port5

; CHECK-INST: sbis 4,  3
; CHECK-INST: sbis 6,  2
; CHECK-INST: sbis 16, 5
; CHECK-INST: sbis 0,  0
; CHECK-INST: sbis 31, 0
; CHECK-INST: sbis 0,  7
; CHECK-INST: sbis 31, 7

; CHECK-INST: sbis 0, 7
