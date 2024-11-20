; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s | llvm-objdump -dr - | FileCheck --check-prefix=CHECK-INST %s

foo:
  and r0,  r15
  and r15, r0
  and r16, r31
  and r31, r16

; CHECK: and r0,  r15               ; encoding: [0x0f,0x20]
; CHECK: and r15, r0                ; encoding: [0xf0,0x20]
; CHECK: and r16, r31               ; encoding: [0x0f,0x23]
; CHECK: and r31, r16               ; encoding: [0xf0,0x23]

; CHECK-INST: and r0,  r15
; CHECK-INST: and r15, r0
; CHECK-INST: and r16, r31
; CHECK-INST: and r31, r16
