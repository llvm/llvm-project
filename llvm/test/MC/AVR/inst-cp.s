; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s | llvm-objdump -d - | FileCheck --check-prefix=CHECK-INST %s


foo:

  cp r12, r2
  cp r19, r0
  cp r15, r31
  cp r0,  r0

; CHECK: cp r12, r2                   ; encoding: [0xc2,0x14]
; CHECK: cp r19, r0                   ; encoding: [0x30,0x15]
; CHECK: cp r15, r31                  ; encoding: [0xff,0x16]
; CHECK: cp r0,  r0                   ; encoding: [0x00,0x14]

; CHECK-INST: cp r12, r2
; CHECK-INST: cp r19, r0
; CHECK-INST: cp r15, r31
; CHECK-INST: cp r0,  r0
