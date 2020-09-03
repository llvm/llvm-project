; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s | llvm-objdump -d - | FileCheck --check-prefix=CHECK-INST %s


foo:
  sub r0,  r15
  sub r15, r0
  sub r16, r31
  sub r31, r16

; CHECK: sub r0,  r15               ; encoding: [0x0f,0x18]
; CHECK: sub r15, r0                ; encoding: [0xf0,0x18]
; CHECK: sub r16, r31               ; encoding: [0x0f,0x1b]
; CHECK: sub r31, r16               ; encoding: [0xf0,0x1b]

; CHECK-INST: sub r0,  r15
; CHECK-INST: sub r15, r0
; CHECK-INST: sub r16, r31
; CHECK-INST: sub r31, r16
