; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s | llvm-objdump -d - | FileCheck --check-prefix=CHECK-INST %s


foo:

  rol r31
  rol r25
  rol r5
  rol r0

; CHECK: rol r31                ; encoding: [0xff,0x1f]
; CHECK: rol r25                ; encoding: [0x99,0x1f]
; CHECK: rol r5                 ; encoding: [0x55,0x1c]
; CHECK: rol r0                 ; encoding: [0x00,0x1c]

; CHECK-INST: rol r31
; CHECK-INST: rol r25
; CHECK-INST: rol r5
; CHECK-INST: rol r0
