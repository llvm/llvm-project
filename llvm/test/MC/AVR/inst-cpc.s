; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s | llvm-objdump -d - | FileCheck --check-prefix=CHECK-INST %s


foo:

  cp r13, r12
  cp r20, r0
  cp r10, r31
  cp r0,  r0

; CHECK: cp r13, r12                  ; encoding: [0xdc,0x14]
; CHECK: cp r20, r0                   ; encoding: [0x40,0x15]
; CHECK: cp r10, r31                  ; encoding: [0xaf,0x16]
; CHECK: cp r0,  r0                   ; encoding: [0x00,0x14]

; CHECK-INST: cp r13, r12
; CHECK-INST: cp r20, r0
; CHECK-INST: cp r10, r31
; CHECK-INST: cp r0,  r0
