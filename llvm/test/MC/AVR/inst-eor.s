; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s | llvm-objdump -d - | FileCheck --check-prefix=CHECK-INST %s


foo:

  eor r0,  r15
  eor r15, r0
  eor r16, r31
  eor r31, r16

; CHECK: eor r0,  r15               ; encoding: [0x0f,0x24]
; CHECK: eor r15, r0                ; encoding: [0xf0,0x24]
; CHECK: eor r16, r31               ; encoding: [0x0f,0x27]
; CHECK: eor r31, r16               ; encoding: [0xf0,0x27]

; CHECK-INST: eor r0,  r15
; CHECK-INST: eor r15, r0
; CHECK-INST: eor r16, r31
; CHECK-INST: eor r31, r16
