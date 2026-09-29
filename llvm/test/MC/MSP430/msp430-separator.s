; RUN: llvm-mc -triple msp430 < %s | FileCheck %s

; MSP430 supports multiple assembly statements on the same line
; separated by a '{' character.

; Check that the '{' is recognized as a line separator and
; multiple statements correctly parsed.

_foo:
; CHECK:      foo
; CHECK:      add r10, r11
; CHECK-NEXT: call r11
; CHECK-NEXT: mov r11, 2(r1)
add r10, r11 { call r11 { mov r11, 2(r1)
ret
