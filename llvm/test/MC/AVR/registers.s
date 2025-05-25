; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s | llvm-objdump -dr - | FileCheck -check-prefix=CHECK-INST %s

; Test register aliases: the upper 6 registers have aliases that can be used in
; assembly.

foo:
  inc xl
  inc xh
  inc yl
  inc yh
  inc zl
  inc zh

  inc XL ; test uppercase

; CHECK: inc r26                    ; encoding: [0xa3,0x95]
; CHECK: inc r27                    ; encoding: [0xb3,0x95]
; CHECK: inc r28                    ; encoding: [0xc3,0x95]
; CHECK: inc r29                    ; encoding: [0xd3,0x95]
; CHECK: inc r30                    ; encoding: [0xe3,0x95]
; CHECK: inc r31                    ; encoding: [0xf3,0x95]

; CHECK: inc r26                    ; encoding: [0xa3,0x95]

; CHECK-INST: inc r26
; CHECK-INST: inc r27
; CHECK-INST: inc r28
; CHECK-INST: inc r29
; CHECK-INST: inc r30
; CHECK-INST: inc r31

; CHECK-INST: inc r26
