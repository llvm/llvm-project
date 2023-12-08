; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s | llvm-objdump -d - | FileCheck --check-prefix=CHECK-INST %s


foo:

  cpse r2, r13
  cpse r9, r0
  cpse r5, r31
  cpse r3, r3

; CHECK: cpse r2, r13                  ; encoding: [0x2d,0x10]
; CHECK: cpse r9, r0                   ; encoding: [0x90,0x10]
; CHECK: cpse r5, r31                  ; encoding: [0x5f,0x12]
; CHECK: cpse r3, r3                   ; encoding: [0x33,0x10]

; CHECK-INST: cpse r2, r13
; CHECK-INST: cpse r9, r0
; CHECK-INST: cpse r5, r31
; CHECK-INST: cpse r3, r3
