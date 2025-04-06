; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s | llvm-objdump -dr - | FileCheck --check-prefix=CHECK-INST %s

foo:
  adc r0,  r15
  adc r15, r0
  adc r16, r31
  adc r31, r16

; CHECK: adc r0,  r15               ; encoding: [0x0f,0x1c]
; CHECK: adc r15, r0                ; encoding: [0xf0,0x1c]
; CHECK: adc r16, r31               ; encoding: [0x0f,0x1f]
; CHECK: adc r31, r16               ; encoding: [0xf0,0x1f]

; CHECK-INST: adc r0,  r15
; CHECK-INST: adc r15, r0
; CHECK-INST: adc r16, r31
; CHECK-INST: adc r31, r16
