; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s


foo:

  cpse r2, r13
  cpse r9, r0
  cpse r5, r31
  cpse r3, r3

; CHECK: cpse r2, r13                  ; encoding: [0x2d,0x10]
; CHECK: cpse r9, r0                   ; encoding: [0x90,0x10]
; CHECK: cpse r5, r31                  ; encoding: [0x5f,0x12]
; CHECK: cpse r3, r3                   ; encoding: [0x33,0x10]
