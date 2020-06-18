; RUN: llvm-mc -triple avr -show-encoding < %s | FileCheck %s
; RUN: llvm-mc -filetype=obj -triple avr < %s | llvm-objdump -d - | FileCheck --check-prefix=CHECK-INST %s


foo:

  mov r2, r13
  mov r9, r0
  mov r5, r31
  mov r3, r3

; CHECK: mov r2, r13                  ; encoding: [0x2d,0x2c]
; CHECK: mov r9, r0                   ; encoding: [0x90,0x2c]
; CHECK: mov r5, r31                  ; encoding: [0x5f,0x2e]
; CHECK: mov r3, r3                   ; encoding: [0x33,0x2c]

; CHECK-INST: mov r2, r13
; CHECK-INST: mov r9, r0
; CHECK-INST: mov r5, r31
; CHECK-INST: mov r3, r3
