; RUN: llvm-mc -triple avr -mattr=des -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr -mattr=des < %s | llvm-objdump --no-print-imm-hex -dr --mattr=des - | FileCheck --check-prefix=CHECK-INST %s

foo:
  des 0
  des 6
  des 1
  des 8

; CHECK: des 0                  ; encoding: [0x0b,0x94]
; CHECK: des 6                  ; encoding: [0x6b,0x94]
; CHECK: des 1                  ; encoding: [0x1b,0x94]
; CHECK: des 8                  ; encoding: [0x8b,0x94]

; CHECK-INST: des 0
; CHECK-INST: des 6
; CHECK-INST: des 1
; CHECK-INST: des 8
