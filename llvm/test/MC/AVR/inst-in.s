; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s | llvm-objdump --no-print-imm-hex -d - | FileCheck -check-prefix=CHECK-INST %s


foo:

  in r2, 4
  in r9, 6
  in r5, 32
  in r0, 0
  in r31, 0
  in r0, 63
  in r31, 63

  in r20, foo+1

; CHECK: in r2, 4                   ; encoding: [0x24,0xb0]
; CHECK: in r9, 6                   ; encoding: [0x96,0xb0]
; CHECK: in r5, 32                  ; encoding: [0x50,0xb4]
; CHECK: in r0, 0                   ; encoding: [0x00,0xb0]
; CHECK: in r31, 0                  ; encoding: [0xf0,0xb1]
; CHECK: in r0, 63                  ; encoding: [0x0f,0xb6]
; CHECK: in r31, 63                 ; encoding: [0xff,0xb7]

; CHECK: in r20, foo+1              ; encoding: [0x40'A',0xb1'A']
; CHECK:                            ;   fixup A - offset: 0, value: foo+1, kind: fixup_port6

; CHECK-INST: in r2, 4
; CHECK-INST: in r9, 6
; CHECK-INST: in r5, 32
; CHECK-INST: in r0, 0
; CHECK-INST: in r31, 0
; CHECK-INST: in r0, 63
; CHECK-INST: in r31, 63

; CHECK-INST: in r20, 0
