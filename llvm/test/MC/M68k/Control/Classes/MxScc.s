; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

; CHECK:      st  %d0
; CHECK-SAME: encoding: [0x50,0xc0]
st	%d0
; CHECK:      sf  %d1
; CHECK-SAME: encoding: [0x51,0xc1]
sf	%d1
; CHECK:      shi  %d2
; CHECK-SAME: encoding: [0x52,0xc2]
shi	%d2
; CHECK:      sls  %d3
; CHECK-SAME: encoding: [0x53,0xc3]
sls	%d3
; CHECK:      scc  %d4
; CHECK-SAME: encoding: [0x54,0xc4]
scc	%d4
; CHECK:      scs  %d5
; CHECK-SAME: encoding: [0x55,0xc5]
scs	%d5
; CHECK:      sne  %d6
; CHECK-SAME: encoding: [0x56,0xc6]
sne	%d6
; CHECK:      seq  %d7
; CHECK-SAME: encoding: [0x57,0xc7]
seq	%d7
; CHECK:      svc  %d0
; CHECK-SAME: encoding: [0x58,0xc0]
svc	%d0
; CHECK:      svs  %d0
; CHECK-SAME: encoding: [0x59,0xc0]
svs	%d0
; CHECK:      spl  %d0
; CHECK-SAME: encoding: [0x5a,0xc0]
spl	%d0
; CHECK:      smi  %d0
; CHECK-SAME: encoding: [0x5b,0xc0]
smi	%d0
; CHECK:      sge  %d0
; CHECK-SAME: encoding: [0x5c,0xc0]
sge	%d0
; CHECK:      slt  %d0
; CHECK-SAME: encoding: [0x5d,0xc0]
slt	%d0
; CHECK:      sgt  %d0
; CHECK-SAME: encoding: [0x5e,0xc0]
sgt	%d0
; CHECK:      sle  %d0
; CHECK-SAME: encoding: [0x5f,0xc0]
sle	%d0

; CHECK:      st  (-1,%a0)
; CHECK-SAME: encoding: [0x50,0xe8,0xff,0xff]
st	(-1,%a0)
; CHECK:      sf  (42,%a1)
; CHECK-SAME: encoding: [0x51,0xe9,0x00,0x2a]
sf	(42,%a1)
; CHECK:      shi  (0,%a2)
; CHECK-SAME: encoding: [0x52,0xea,0x00,0x00]
shi	(0,%a2)
; CHECK:      sls  (0,%a3)
; CHECK-SAME: encoding: [0x53,0xeb,0x00,0x00]
sls	(0,%a3)
; CHECK:      scc  (0,%a4)
; CHECK-SAME: encoding: [0x54,0xec,0x00,0x00]
scc	(0,%a4)
; CHECK:      scs  (0,%a5)
; CHECK-SAME: encoding: [0x55,0xed,0x00,0x00]
scs	(0,%a5)
; CHECK:      sne  (0,%a6)
; CHECK-SAME: encoding: [0x56,0xee,0x00,0x00]
sne	(0,%a6)
; CHECK:      seq  (0,%a0)
; CHECK-SAME: encoding: [0x57,0xe8,0x00,0x00]
seq	(0,%a0)
; CHECK:      svc  (0,%a0)
; CHECK-SAME: encoding: [0x58,0xe8,0x00,0x00]
svc	(0,%a0)
; CHECK:      svs  (0,%a0)
; CHECK-SAME: encoding: [0x59,0xe8,0x00,0x00]
svs	(0,%a0)
; CHECK:      spl  (0,%a0)
; CHECK-SAME: encoding: [0x5a,0xe8,0x00,0x00]
spl	(0,%a0)
; CHECK:      smi  (0,%a0)
; CHECK-SAME: encoding: [0x5b,0xe8,0x00,0x00]
smi	(0,%a0)
; CHECK:      sge  (0,%a0)
; CHECK-SAME: encoding: [0x5c,0xe8,0x00,0x00]
sge	(0,%a0)
; CHECK:      slt  (0,%a0)
; CHECK-SAME: encoding: [0x5d,0xe8,0x00,0x00]
slt	(0,%a0)
; CHECK:      sgt  (0,%a0)
; CHECK-SAME: encoding: [0x5e,0xe8,0x00,0x00]
sgt	(0,%a0)
; CHECK:      sle  (0,%a0)
; CHECK-SAME: encoding: [0x5f,0xe8,0x00,0x00]
sle	(0,%a0)

