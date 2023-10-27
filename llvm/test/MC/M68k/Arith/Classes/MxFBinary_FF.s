; RUN: llvm-mc -triple=m68k -assemble -show-encoding -mcpu=M68040 %s | FileCheck %s

; CHECK: fadd.x  %fp0, %fp1
; CHECK-SAME: ; encoding: [0xf2,0x00,0x00,0xa2]
fadd.x %fp0, %fp1

; CHECK: fsadd.x %fp2, %fp3
; CHECK-SAME: ; encoding: [0xf2,0x00,0x09,0xe2]
fsadd.x %fp2, %fp3

; CHECK: fdadd.x %fp3, %fp4
; CHECK-SAME: ; encoding: [0xf2,0x00,0x0e,0x66]
fdadd.x %fp3, %fp4

; CHECK: fsub.x  %fp1, %fp2
; CHECK-SAME: ; encoding: [0xf2,0x00,0x05,0x28]
fsub.x %fp1, %fp2

; CHECK: fssub.x %fp3, %fp4
; CHECK-SAME; encoding: [0xf2,0x00,0x0e,0x68]
fssub.x %fp3, %fp4

; CHECK: fdsub.x %fp5, %fp6
; CHECK-SAME; encoding: [0xf2,0x00,0x17,0x6c]
fdsub.x %fp5, %fp6

; CHECK: fmul.x  %fp2, %fp3
; CHECK-SAME; encoding: [0xf2,0x00,0x09,0xa3]
fmul.x %fp2, %fp3

; CHECK: fsmul.x %fp4, %fp5
; CHECK-SAME; encoding: [0xf2,0x00,0x12,0xe3]
fsmul.x %fp4, %fp5

; CHECK: fdmul.x %fp6, %fp7
; CHECK-SAME; encoding: [0xf2,0x00,0x1b,0xe7]
fdmul.x %fp6, %fp7

; CHECK: fdiv.x  %fp3, %fp4
; CHECK-SAME; encoding: [0xf2,0x00,0x0e,0x20]
fdiv.x %fp3, %fp4

; CHECK: fsdiv.x %fp5, %fp6
; CHECK-SAME; encoding: [0xf2,0x00,0x17,0x60]
fsdiv.x %fp5, %fp6

; CHECK: fddiv.x %fp7, %fp0
; CHECK-SAME; encoding: [0xf2,0x00,0x1c,0x64]
fddiv.x %fp7, %fp0
