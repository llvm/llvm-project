; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

; CHECK:      btst  %d0, (-1,%pc,%d1)
; CHECK-SAME: encoding: [0x01,0x3b,0x18,0xff]
btst	%d0, (-1,%pc,%d1)
; CHECK:      btst  %d1, (0,%pc,%d0)
; CHECK-SAME: encoding: [0x03,0x3b,0x08,0x00]
btst	%d1, (0,%pc,%d0)

; CHECK:      btst  %d0, (0,%pc)
; CHECK-SAME: encoding: [0x01,0x3a,0x00,0x00]
btst	%d0, (0,%pc)
; CHECK:      btst  %d1, (-1,%pc)
; CHECK-SAME: encoding: [0x03,0x3a,0xff,0xff]
btst	%d1, (-1,%pc)

; CHECK:      btst  %d0, (-1,%a1,%a0)
; CHECK-SAME: encoding: [0x01,0x31,0x88,0xff]
btst	%d0, (-1,%a1,%a0)
; CHECK:      btst  %d1, (0,%a0,%a0)
; CHECK-SAME: encoding: [0x03,0x30,0x88,0x00]
btst	%d1, (0,%a0,%a0)

; CHECK:      btst  %d0, (-1,%a1)
; CHECK-SAME: encoding: [0x01,0x29,0xff,0xff]
btst	%d0, (-1,%a1)
; CHECK:      btst  %d1, (0,%a0)
; CHECK-SAME: encoding: [0x03,0x28,0x00,0x00]
btst	%d1, (0,%a0)

; CHECK:      btst  %d0, (%a1)
; CHECK-SAME: encoding: [0x01,0x11]
btst	%d0, (%a1)
; CHECK:      btst  %d1, (%a0)
; CHECK-SAME: encoding: [0x03,0x10]
btst	%d1, (%a0)

