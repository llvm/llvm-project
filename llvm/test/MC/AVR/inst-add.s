; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s | llvm-objdump -d - | FileCheck --check-prefix=CHECK-INST %s


foo:

  add r0,  r15
  add r15, r0
  add r16, r31
  add r31, r16

; CHECK: add r0,  r15               ; encoding: [0x0f,0x0c]
; CHECK: add r15, r0                ; encoding: [0xf0,0x0c]
; CHECK: add r16, r31               ; encoding: [0x0f,0x0f]
; CHECK: add r31, r16               ; encoding: [0xf0,0x0f]

; CHECK-INST: add r0,  r15
; CHECK-INST: add r15, r0
; CHECK-INST: add r16, r31
; CHECK-INST: add r31, r16
