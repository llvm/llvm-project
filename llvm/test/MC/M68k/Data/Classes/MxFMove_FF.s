; RUN: llvm-mc -triple=m68k -mcpu=M68040 -show-encoding < %s | FileCheck %s

; CHECK: fmove.x %fp7, %fp2
; CHECK-SAME: [0xf2,0x00,0x1d,0x00]
fmove.x %fp7, %fp2

; CHECK: fsmove.x %fp6, %fp1
; CHECK-SAME: [0xf2,0x00,0x18,0xc0]
fsmove.x %fp6, %fp1

; CHECK: fdmove.x %fp3, %fp0
; CHECK-SAME: [0xf2,0x00,0x0c,0x44]
fdmove.x %fp3, %fp0
