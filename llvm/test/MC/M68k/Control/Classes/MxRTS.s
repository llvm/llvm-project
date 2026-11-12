; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

; CHECK:      rts
; CHECK-SAME: encoding: [0x4e,0x75]
rts
