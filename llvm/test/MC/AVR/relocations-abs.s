; RUN: llvm-mc -filetype=obj -triple=avr %s | llvm-objdump -dr -r - | FileCheck %s

; CHECK: <bar>:
; CHECK-NEXT: 00 00 nop
; CHECK-NEXT: R_AVR_16 .text+0x2
bar:
    .short 1f
1:
