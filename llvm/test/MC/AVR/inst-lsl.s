; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s | llvm-objdump -d - | FileCheck --check-prefix=CHECK-INST %s


foo:

  lsl r31
  lsl r25
  lsl r5
  lsl r0

; CHECK: lsl r31                ; encoding: [0xff,0x0f]
; CHECK: lsl r25                ; encoding: [0x99,0x0f]
; CHECK: lsl r5                 ; encoding: [0x55,0x0c]
; CHECK: lsl r0                 ; encoding: [0x00,0x0c]

; CHECK-INST: lsl r31
; CHECK-INST: lsl r25
; CHECK-INST: lsl r5
; CHECK-INST: lsl r0
