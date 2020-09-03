; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s | llvm-objdump -d - | FileCheck -check-prefix=CHECK-INST %s


foo:

  inc r12
  inc r29
  inc r6
  inc r20
  inc r0
  inc r31

; CHECK: inc r12                  ; encoding: [0xc3,0x94]
; CHECK: inc r29                  ; encoding: [0xd3,0x95]
; CHECK: inc r6                   ; encoding: [0x63,0x94]
; CHECK: inc r20                  ; encoding: [0x43,0x95]
; CHECK: inc r0                   ; encoding: [0x03,0x94]
; CHECK: inc r31                  ; encoding: [0xf3,0x95]

; CHECK-INST: inc r12
; CHECK-INST: inc r29
; CHECK-INST: inc r6
; CHECK-INST: inc r20
; CHECK-INST: inc r0
; CHECK-INST: inc r31
