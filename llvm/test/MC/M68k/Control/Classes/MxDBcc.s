; RUN: llvm-mc -triple=m68k -motorola-integers -show-encoding %s | FileCheck %s

; CHECK:      dbt  %d0, $1
; CHECK-SAME: encoding: [0x50,0xc8,0x00,0x01]
dbt	%d0, $1

; CHECK:      dbcc  %d4, $28
; CHECK-SAME: encoding: [0x54,0xcc,0x00,0x28]
dbcc	%d4, $28
