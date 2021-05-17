; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

; CHECK:      nop
; CHECK-SAME: encoding: [0x4e,0x71]
nop

