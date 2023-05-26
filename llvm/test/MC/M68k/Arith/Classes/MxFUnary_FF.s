; RUN: llvm-mc -triple=m68k -assemble -show-encoding -mcpu=M68040 %s | FileCheck %s

; CHECK: fabs.x  %fp3, %fp2
; CHECK-SAME: ; encoding: [0xf2,0x00,0x0d,0x18]
fabs.x %fp3, %fp2

; CHECK: fsabs.x %fp5, %fp7
; CHECK-SAME: ; encoding: [0xf2,0x00,0x17,0xd8]
fsabs.x %fp5, %fp7

; CHECK: fdabs.x %fp0, %fp0
; CHECK-SAME: ; encoding: [0xf2,0x00,0x00,0x5c]
fdabs.x %fp0, %fp0

; CHECK: fneg.x  %fp0, %fp1
; CHECK-SAME: ; encoding: [0xf2,0x00,0x00,0x9a]
fneg.x %fp0, %fp1

; CHECK: fsneg.x %fp2, %fp3
; CHECK-SAME: ; encoding: [0xf2,0x00,0x09,0xda]
fsneg.x %fp2, %fp3

; CHECK: fdneg.x %fp4, %fp1
; CHECK-SAME: ; encoding: [0xf2,0x00,0x10,0xde]
fdneg.x %fp4, %fp1
