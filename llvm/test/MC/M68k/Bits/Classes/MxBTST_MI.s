; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

; CHECK:      bchg  #0, (%a1)
; CHECK-SAME: encoding: [0x08,0x51,0x00,0x00]
bchg	#0, (%a1)
; CHECK:      bchg  #-1, (%a0)
; CHECK-SAME: encoding: [0x08,0x50,0x00,0xff]
bchg	#-1, (%a0)

; CHECK:      bclr  #0, (%a1)
; CHECK-SAME: encoding: [0x08,0x91,0x00,0x00]
bclr	#0, (%a1)
; CHECK:      bclr  #-1, (%a0)
; CHECK-SAME: encoding: [0x08,0x90,0x00,0xff]
bclr	#-1, (%a0)

; CHECK:      bset  #0, (%a1)
; CHECK-SAME: encoding: [0x08,0xd1,0x00,0x00]
bset	#0, (%a1)
; CHECK:      bset  #-1, (%a0)
; CHECK-SAME: encoding: [0x08,0xd0,0x00,0xff]
bset	#-1, (%a0)

; CHECK:      btst  #0, (%a1)
; CHECK-SAME: encoding: [0x08,0x11,0x00,0x00]
btst	#0, (%a1)
; CHECK:      btst  #-1, (%a0)
; CHECK-SAME: encoding: [0x08,0x10,0x00,0xff]
btst	#-1, (%a0)

; CHECK:      bchg  #0, (%a1)+
; CHECK-SAME: encoding: [0x08,0x59,0x00,0x00]
bchg	#0, (%a1)+
; CHECK:      bchg  #-1, (%a0)+
; CHECK-SAME: encoding: [0x08,0x58,0x00,0xff]
bchg	#-1, (%a0)+

; CHECK:      bclr  #0, (%a1)+
; CHECK-SAME: encoding: [0x08,0x99,0x00,0x00]
bclr	#0, (%a1)+
; CHECK:      bclr  #-1, (%a0)+
; CHECK-SAME: encoding: [0x08,0x98,0x00,0xff]
bclr	#-1, (%a0)+

; CHECK:      bset  #0, (%a1)+
; CHECK-SAME: encoding: [0x08,0xd9,0x00,0x00]
bset	#0, (%a1)+
; CHECK:      bset  #-1, (%a0)+
; CHECK-SAME: encoding: [0x08,0xd8,0x00,0xff]
bset	#-1, (%a0)+

; CHECK:      btst  #0, (%a1)+
; CHECK-SAME: encoding: [0x08,0x19,0x00,0x00]
btst	#0, (%a1)+
; CHECK:      btst  #-1, (%a0)+
; CHECK-SAME: encoding: [0x08,0x18,0x00,0xff]
btst	#-1, (%a0)+

; CHECK:      bchg  #0, -(%a1)
; CHECK-SAME: encoding: [0x08,0x61,0x00,0x00]
bchg	#0, -(%a1)
; CHECK:      bchg  #-1, -(%a0)
; CHECK-SAME: encoding: [0x08,0x60,0x00,0xff]
bchg	#-1, -(%a0)

; CHECK:      bclr  #0, -(%a1)
; CHECK-SAME: encoding: [0x08,0xa1,0x00,0x00]
bclr	#0, -(%a1)
; CHECK:      bclr  #-1, -(%a0)
; CHECK-SAME: encoding: [0x08,0xa0,0x00,0xff]
bclr	#-1, -(%a0)

; CHECK:      bset  #0, -(%a1)
; CHECK-SAME: encoding: [0x08,0xe1,0x00,0x00]
bset	#0, -(%a1)
; CHECK:      bset  #-1, -(%a0)
; CHECK-SAME: encoding: [0x08,0xe0,0x00,0xff]
bset	#-1, -(%a0)

; CHECK:      btst  #0, -(%a1)
; CHECK-SAME: encoding: [0x08,0x21,0x00,0x00]
btst	#0, -(%a1)
; CHECK:      btst  #-1, -(%a0)
; CHECK-SAME: encoding: [0x08,0x20,0x00,0xff]
btst	#-1, -(%a0)

; CHECK:      bchg  #0, (-1,%a1)
; CHECK-SAME: encoding: [0x08,0x69,0x00,0x00,0xff,0xff]
bchg	#0, (-1,%a1)
; CHECK:      bchg  #-1, (0,%a0)
; CHECK-SAME: encoding: [0x08,0x68,0x00,0xff,0x00,0x00]
bchg	#-1, (0,%a0)

; CHECK:      bclr  #0, (-1,%a1)
; CHECK-SAME: encoding: [0x08,0xa9,0x00,0x00,0xff,0xff]
bclr	#0, (-1,%a1)
; CHECK:      bclr  #-1, (0,%a0)
; CHECK-SAME: encoding: [0x08,0xa8,0x00,0xff,0x00,0x00]
bclr	#-1, (0,%a0)

; CHECK:      bset  #0, (-1,%a1)
; CHECK-SAME: encoding: [0x08,0xe9,0x00,0x00,0xff,0xff]
bset	#0, (-1,%a1)
; CHECK:      bset  #-1, (0,%a0)
; CHECK-SAME: encoding: [0x08,0xe8,0x00,0xff,0x00,0x00]
bset	#-1, (0,%a0)

; CHECK:      btst  #0, (-1,%a1)
; CHECK-SAME: encoding: [0x08,0x29,0x00,0x00,0xff,0xff]
btst	#0, (-1,%a1)
; CHECK:      btst  #-1, (0,%a0)
; CHECK-SAME: encoding: [0x08,0x28,0x00,0xff,0x00,0x00]
btst	#-1, (0,%a0)

; CHECK:      bchg  #0, (-1,%a1,%a0)
; CHECK-SAME: encoding: [0x08,0x71,0x00,0x00,0x88,0xff]
bchg	#0, (-1,%a1,%a0)
; CHECK:      bchg  #-1, (0,%a0,%a0)
; CHECK-SAME: encoding: [0x08,0x70,0x00,0xff,0x88,0x00]
bchg	#-1, (0,%a0,%a0)

; CHECK:      bclr  #0, (-1,%a1,%a0)
; CHECK-SAME: encoding: [0x08,0xb1,0x00,0x00,0x88,0xff]
bclr	#0, (-1,%a1,%a0)
; CHECK:      bclr  #-1, (0,%a0,%a0)
; CHECK-SAME: encoding: [0x08,0xb0,0x00,0xff,0x88,0x00]
bclr	#-1, (0,%a0,%a0)

; CHECK:      bset  #0, (-1,%a1,%a0)
; CHECK-SAME: encoding: [0x08,0xf1,0x00,0x00,0x88,0xff]
bset	#0, (-1,%a1,%a0)
; CHECK:      bset  #-1, (0,%a0,%a0)
; CHECK-SAME: encoding: [0x08,0xf0,0x00,0xff,0x88,0x00]
bset	#-1, (0,%a0,%a0)

; CHECK:      btst  #0, (-1,%a1,%a0)
; CHECK-SAME: encoding: [0x08,0x31,0x00,0x00,0x88,0xff]
btst	#0, (-1,%a1,%a0)
; CHECK:      btst  #-1, (0,%a0,%a0)
; CHECK-SAME: encoding: [0x08,0x30,0x00,0xff,0x88,0x00]
btst	#-1, (0,%a0,%a0)

; CHECK:      btst  #0, (0,%pc)
; CHECK-SAME: encoding: [0x08,0x3a,0x00,0x00,0x00,0x00]
btst	#0, (0,%pc)
; CHECK:      btst  #-1, (-1,%pc)
; CHECK-SAME: encoding: [0x08,0x3a,0x00,0xff,0xff,0xff]
btst	#-1, (-1,%pc)

; CHECK:      btst  #0, (-1,%pc,%d1)
; CHECK-SAME: encoding: [0x08,0x3b,0x00,0x00,0x18,0xff]
btst	#0, (-1,%pc,%d1)
; CHECK:      btst  #1, (0,%pc,%d0)
; CHECK-SAME: encoding: [0x08,0x3b,0x00,0x01,0x08,0x00]
btst	#1, (0,%pc,%d0)
