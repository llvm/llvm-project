; RUN: llvm-mc -filetype=obj -triple avr < %s | llvm-objdump -d - | FileCheck %s

foo:

  ; The $ symbol is a separator (like a newline).
  mov r0, r1 $ mov r1, r2 $ mov r2, r3 $ mov r3, r4

; CHECK: mov r0, r1
; CHECK: mov r1, r2
; CHECK: mov r2, r3
; CHECK: mov r3, r4
