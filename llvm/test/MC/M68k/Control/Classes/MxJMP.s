; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

; CHECK:      jmp  (%a0)
; CHECK-SAME: encoding: [0x4e,0xd0]
jmp	(%a0)

