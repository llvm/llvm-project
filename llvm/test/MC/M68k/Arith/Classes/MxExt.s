; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

; CHECK:      ext.w  %d0
; CHECK-SAME: encoding: [0x48,0x80]
ext.w	%d0
; CHECK:      ext.w  %d3
; CHECK-SAME: encoding: [0x48,0x83]
ext.w	%d3
; CHECK:      ext.l  %d0
; CHECK-SAME: encoding: [0x48,0xc0]
ext.l	%d0
; CHECK:      ext.l  %d7
; CHECK-SAME: encoding: [0x48,0xc7]
ext.l	%d7

