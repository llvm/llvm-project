; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s | llvm-objdump --no-print-imm-hex -d - | FileCheck -check-prefix=CHECK-INST %s


foo:

  sbi 3, 5
  sbi 1, 1
  sbi 7, 2
  sbi 0, 0
  sbi 0, 7
  sbi 31, 0
  sbi 31, 7

  sbi main, 0

; CHECK: sbi 3, 5                  ; encoding: [0x1d,0x9a]
; CHECK: sbi 1, 1                  ; encoding: [0x09,0x9a]
; CHECK: sbi 7, 2                  ; encoding: [0x3a,0x9a]
; CHECK: sbi 0, 0                  ; encoding: [0x00,0x9a]
; CHECK: sbi 0, 7                  ; encoding: [0x07,0x9a]
; CHECK: sbi 31, 0                 ; encoding: [0xf8,0x9a]
; CHECK: sbi 31, 7                 ; encoding: [0xff,0x9a]

; CHECK: sbi main, 0               ; encoding: [0bAAAAA000,0x9a]
; CHECK:                           ;   fixup A - offset: 0, value: main, kind: fixup_port5

; CHECK-INST: sbi 3, 5
; CHECK-INST: sbi 1, 1
; CHECK-INST: sbi 7, 2
; CHECK-INST: sbi 0, 0
; CHECK-INST: sbi 0, 7
; CHECK-INST: sbi 31, 0
; CHECK-INST: sbi 31, 7

; CHECK-INST: sbi 0, 0
