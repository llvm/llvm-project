; RUN: llc < %s -march=xcore | FileCheck %s

@size = global i32 0		; <i32*> [#uses=1]
@g0 = external global i32		; <i32*> [#uses=2]
@g1 = external global i32		; <i32*> [#uses=2]
@g2 = external global i32		; <i32*> [#uses=2]
@g3 = external global i32		; <i32*> [#uses=2]
@g4 = external global i32		; <i32*> [#uses=2]
@g5 = external global i32		; <i32*> [#uses=2]
@g6 = external global i32		; <i32*> [#uses=2]
@g7 = external global i32		; <i32*> [#uses=2]
@g8 = external global i32		; <i32*> [#uses=2]
@g9 = external global i32		; <i32*> [#uses=2]
@g10 = external global i32		; <i32*> [#uses=2]
@g11 = external global i32		; <i32*> [#uses=2]

define void @f() nounwind {
entry:
	%x = alloca [100 x i32], align 4		; <[100 x i32]*> [#uses=2]
	%0 = load i32, i32* @size, align 4		; <i32> [#uses=1]
	%1 = alloca i32, i32 %0, align 4		; <i32*> [#uses=1]
	%2 = load volatile i32, i32* @g0, align 4		; <i32> [#uses=1]
	%3 = load volatile i32, i32* @g1, align 4		; <i32> [#uses=1]
	%4 = load volatile i32, i32* @g2, align 4		; <i32> [#uses=1]
	%5 = load volatile i32, i32* @g3, align 4		; <i32> [#uses=1]
	%6 = load volatile i32, i32* @g4, align 4		; <i32> [#uses=1]
	%7 = load volatile i32, i32* @g5, align 4		; <i32> [#uses=1]
	%8 = load volatile i32, i32* @g6, align 4		; <i32> [#uses=1]
	%9 = load volatile i32, i32* @g7, align 4		; <i32> [#uses=1]
	%10 = load volatile i32, i32* @g8, align 4		; <i32> [#uses=1]
	%11 = load volatile i32, i32* @g9, align 4		; <i32> [#uses=1]
	%12 = load volatile i32, i32* @g10, align 4		; <i32> [#uses=1]
	%13 = load volatile i32, i32* @g11, align 4		; <i32> [#uses=2]
	%14 = getelementptr [100 x i32], [100 x i32]* %x, i32 0, i32 50		; <i32*> [#uses=1]
	store i32 %13, i32* %14, align 4
	store volatile i32 %13, i32* @g11, align 4
	store volatile i32 %12, i32* @g10, align 4
	store volatile i32 %11, i32* @g9, align 4
	store volatile i32 %10, i32* @g8, align 4
	store volatile i32 %9, i32* @g7, align 4
	store volatile i32 %8, i32* @g6, align 4
	store volatile i32 %7, i32* @g5, align 4
	store volatile i32 %6, i32* @g4, align 4
	store volatile i32 %5, i32* @g3, align 4
	store volatile i32 %4, i32* @g2, align 4
	store volatile i32 %3, i32* @g1, align 4
	store volatile i32 %2, i32* @g0, align 4
	%x1 = getelementptr [100 x i32], [100 x i32]* %x, i32 0, i32 0		; <i32*> [#uses=1]
	call void @g(i32* %x1, i32* %1) nounwind
	ret void
}
declare void @g(i32*, i32*)


; CHECK: .section .cp.rodata.cst4,"aMc",@progbits,4
; CHECK: .p2align  2
; CHECK: [[ARG5:.LCPI[0-9_]+]]:
; CHECK: .long   100002
; CHECK: [[INDEX0:.LCPI[0-9_]+]]:
; CHECK: .long   100004
; CHECK: [[INDEX1:.LCPI[0-9_]+]]:
; CHECK: .long   80002
; CHECK: [[INDEX2:.LCPI[0-9_]+]]:
; CHECK: .long   81002
; CHECK: [[INDEX3:.LCPI[0-9_]+]]:
; CHECK: .long   82002
; CHECK: [[INDEX4:.LCPI[0-9_]+]]:
; CHECK: .long   83002
; CHECK: [[INDEX5:.LCPI[0-9_]+]]:
; CHECK: .long   84002
; CHECK: .text
; !FP + large frame: spill SR+SR = entsp 2 + 100000
; CHECK-LABEL: ScavengeSlots:
; CHECK: entsp 65535
; CHECK: extsp 34468
; scavenge r11
; CHECK: ldaw r11, sp[0]
; scavenge r4 using SR spill slot
; CHECK: stw r4, sp[1]
; CHECK: ldw r4, cp[[[ARG5]]]
; r11 used to load 5th argument
; CHECK: stw r10, r11[r4]
; CHECK: ldaw r10, sp[0]
; CHECK: ldw r11, cp[[[INDEX0]]]
; CHECK: ldw r10, r10[r11]
; CHECK: ldaw r11, sp[0]
; CHECK: ldw r4, cp[[[INDEX1]]]
; CHECK: stw r0, r11[r4]
; CHECK: ldaw r0, sp[0]
; CHECK: ldw r11, cp[[[INDEX2]]]
; CHECK: stw r1, r0[r11]
; CHECK: ldaw r0, sp[0]
; CHECK: ldw r1, cp[[[INDEX3]]]
; CHECK: stw r3, r0[r1]
; CHECK: ldaw r0, sp[0]
; CHECK: ldw r1, cp[[[INDEX5]]]
; CHECK: stw r10, r0[r1]
; CHECK: ldaw r10, sp[0]
; CHECK: ldw r0, cp[[[ARG5]]]
; CHECK: ldw r10, r10[r0]
; CHECK: ldaw sp, sp[65535]
; CHECK: ldw r4, sp[1]
; CHECK: retsp 34468
define void @ScavengeSlots(i32 %r0, i32 %r1, i32 %r2, i32 %r3, i32 %r4) nounwind {
entry:
  %Data = alloca [100000 x i32]
  %i0 = getelementptr inbounds [100000 x i32], [100000 x i32]* %Data, i32 0, i32 80000
  store volatile i32 %r0, i32* %i0
  %i1 = getelementptr inbounds [100000 x i32], [100000 x i32]* %Data, i32 0, i32 81000
  store volatile i32 %r1, i32* %i1
  %i2 = getelementptr inbounds [100000 x i32], [100000 x i32]* %Data, i32 0, i32 82000
  store volatile i32 %r2, i32* %i2
  %i3 = getelementptr inbounds [100000 x i32], [100000 x i32]* %Data, i32 0, i32 83000
  store volatile i32 %r3, i32* %i3
  %i4 = getelementptr inbounds [100000 x i32], [100000 x i32]* %Data, i32 0, i32 84000
  store volatile i32 %r4, i32* %i4
  ret void
}
