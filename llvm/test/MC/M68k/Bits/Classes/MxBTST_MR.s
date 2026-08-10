; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

; CHECK:      bchg  %d0, (%a1)
; CHECK-SAME: encoding: [0x01,0x51]
bchg	%d0, (%a1)
; CHECK:      bchg  %d1, (%a0)
; CHECK-SAME: encoding: [0x03,0x50]
bchg	%d1, (%a0)

; CHECK:      bclr  %d0, (%a1)
; CHECK-SAME: encoding: [0x01,0x91]
bclr	%d0, (%a1)
; CHECK:      bclr  %d1, (%a0)
; CHECK-SAME: encoding: [0x03,0x90]
bclr	%d1, (%a0)

; CHECK:      bset  %d0, (%a1)
; CHECK-SAME: encoding: [0x01,0xd1]
bset	%d0, (%a1)
; CHECK:      bset  %d1, (%a0)
; CHECK-SAME: encoding: [0x03,0xd0]
bset	%d1, (%a0)

; CHECK:      btst  %d0, (%a1)
; CHECK-SAME: encoding: [0x01,0x11]
btst	%d0, (%a1)
; CHECK:      btst  %d1, (%a0)
; CHECK-SAME: encoding: [0x03,0x10]
btst	%d1, (%a0)

; CHECK:      bchg  %d0, (%a1)+
; CHECK-SAME: encoding: [0x01,0x59]
bchg	%d0, (%a1)+
; CHECK:      bchg  %d1, (%a0)+
; CHECK-SAME: encoding: [0x03,0x58]
bchg	%d1, (%a0)+

; CHECK:      bclr  %d0, (%a1)+
; CHECK-SAME: encoding: [0x01,0x99]
bclr	%d0, (%a1)+
; CHECK:      bclr  %d1, (%a0)+
; CHECK-SAME: encoding: [0x03,0x98]
bclr	%d1, (%a0)+

; CHECK:      bset  %d0, (%a1)+
; CHECK-SAME: encoding: [0x01,0xd9]
bset	%d0, (%a1)+
; CHECK:      bset  %d1, (%a0)+
; CHECK-SAME: encoding: [0x03,0xd8]
bset	%d1, (%a0)+

; CHECK:      btst  %d0, (%a1)+
; CHECK-SAME: encoding: [0x01,0x19]
btst	%d0, (%a1)+
; CHECK:      btst  %d1, (%a0)+
; CHECK-SAME: encoding: [0x03,0x18]
btst	%d1, (%a0)+

; CHECK:      bchg  %d0, -(%a1)
; CHECK-SAME: encoding: [0x01,0x61]
bchg	%d0, -(%a1)
; CHECK:      bchg  %d1, -(%a0)
; CHECK-SAME: encoding: [0x03,0x60]
bchg	%d1, -(%a0)

; CHECK:      bclr  %d0, -(%a1)
; CHECK-SAME: encoding: [0x01,0xa1]
bclr	%d0, -(%a1)
; CHECK:      bclr  %d1, -(%a0)
; CHECK-SAME: encoding: [0x03,0xa0]
bclr	%d1, -(%a0)

; CHECK:      bset  %d0, -(%a1)
; CHECK-SAME: encoding: [0x01,0xe1]
bset	%d0, -(%a1)
; CHECK:      bset  %d1, -(%a0)
; CHECK-SAME: encoding: [0x03,0xe0]
bset	%d1, -(%a0)

; CHECK:      btst  %d0, -(%a1)
; CHECK-SAME: encoding: [0x01,0x21]
btst	%d0, -(%a1)
; CHECK:      btst  %d1, -(%a0)
; CHECK-SAME: encoding: [0x03,0x20]
btst	%d1, -(%a0)

; CHECK:      bchg  %d0, (-1,%a1)
; CHECK-SAME: encoding: [0x01,0x69,0xff,0xff]
bchg	%d0, (-1,%a1)
; CHECK:      bchg  %d1, (0,%a0)
; CHECK-SAME: encoding: [0x03,0x68,0x00,0x00]
bchg	%d1, (0,%a0)

; CHECK:      bclr  %d0, (-1,%a1)
; CHECK-SAME: encoding: [0x01,0xa9,0xff,0xff]
bclr	%d0, (-1,%a1)
; CHECK:      bclr  %d1, (0,%a0)
; CHECK-SAME: encoding: [0x03,0xa8,0x00,0x00]
bclr	%d1, (0,%a0)

; CHECK:      bset  %d0, (-1,%a1)
; CHECK-SAME: encoding: [0x01,0xe9,0xff,0xff]
bset	%d0, (-1,%a1)
; CHECK:      bset  %d1, (0,%a0)
; CHECK-SAME: encoding: [0x03,0xe8,0x00,0x00]
bset	%d1, (0,%a0)

; CHECK:      btst  %d0, (-1,%a1)
; CHECK-SAME: encoding: [0x01,0x29,0xff,0xff]
btst	%d0, (-1,%a1)
; CHECK:      btst  %d1, (0,%a0)
; CHECK-SAME: encoding: [0x03,0x28,0x00,0x00]
btst	%d1, (0,%a0)

; CHECK:      bchg  %d0, (-1,%a1,%a0)
; CHECK-SAME: encoding: [0x01,0x71,0x88,0xff]
bchg	%d0, (-1,%a1,%a0)
; CHECK:      bchg  %d1, (0,%a0,%a0)
; CHECK-SAME: encoding: [0x03,0x70,0x88,0x00]
bchg	%d1, (0,%a0,%a0)

; CHECK:      bclr  %d0, (-1,%a1,%a0)
; CHECK-SAME: encoding: [0x01,0xb1,0x88,0xff]
bclr	%d0, (-1,%a1,%a0)
; CHECK:      bclr  %d1, (0,%a0,%a0)
; CHECK-SAME: encoding: [0x03,0xb0,0x88,0x00]
bclr	%d1, (0,%a0,%a0)

; CHECK:      bset  %d0, (-1,%a1,%a0)
; CHECK-SAME: encoding: [0x01,0xf1,0x88,0xff]
bset	%d0, (-1,%a1,%a0)
; CHECK:      bset  %d1, (0,%a0,%a0)
; CHECK-SAME: encoding: [0x03,0xf0,0x88,0x00]
bset	%d1, (0,%a0,%a0)

; CHECK:      btst  %d0, (-1,%a1,%a0)
; CHECK-SAME: encoding: [0x01,0x31,0x88,0xff]
btst	%d0, (-1,%a1,%a0)
; CHECK:      btst  %d1, (0,%a0,%a0)
; CHECK-SAME: encoding: [0x03,0x30,0x88,0x00]
btst	%d1, (0,%a0,%a0)

; CHECK:      btst  %d0, (0,%pc)
; CHECK-SAME: encoding: [0x01,0x3a,0x00,0x00]
btst	%d0, (0,%pc)
; CHECK:      btst  %d1, (-1,%pc)
; CHECK-SAME: encoding: [0x03,0x3a,0xff,0xff]
btst	%d1, (-1,%pc)

; CHECK:      btst  %d0, (-1,%pc,%d1)
; CHECK-SAME: encoding: [0x01,0x3b,0x18,0xff]
btst	%d0, (-1,%pc,%d1)
; CHECK:      btst  %d1, (0,%pc,%d0)
; CHECK-SAME: encoding: [0x03,0x3b,0x08,0x00]
btst	%d1, (0,%pc,%d0)
