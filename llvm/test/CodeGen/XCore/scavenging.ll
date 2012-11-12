; RUN: llc < %s -march=xcore
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
	%0 = load i32* @size, align 4		; <i32> [#uses=1]
	%1 = alloca i32, i32 %0, align 4		; <i32*> [#uses=1]
	%2 = load volatile i32* @g0, align 4		; <i32> [#uses=1]
	%3 = load volatile i32* @g1, align 4		; <i32> [#uses=1]
	%4 = load volatile i32* @g2, align 4		; <i32> [#uses=1]
	%5 = load volatile i32* @g3, align 4		; <i32> [#uses=1]
	%6 = load volatile i32* @g4, align 4		; <i32> [#uses=1]
	%7 = load volatile i32* @g5, align 4		; <i32> [#uses=1]
	%8 = load volatile i32* @g6, align 4		; <i32> [#uses=1]
	%9 = load volatile i32* @g7, align 4		; <i32> [#uses=1]
	%10 = load volatile i32* @g8, align 4		; <i32> [#uses=1]
	%11 = load volatile i32* @g9, align 4		; <i32> [#uses=1]
	%12 = load volatile i32* @g10, align 4		; <i32> [#uses=1]
	%13 = load volatile i32* @g11, align 4		; <i32> [#uses=2]
	%14 = getelementptr [100 x i32]* %x, i32 0, i32 50		; <i32*> [#uses=1]
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
	%x1 = getelementptr [100 x i32]* %x, i32 0, i32 0		; <i32*> [#uses=1]
	call void @g(i32* %x1, i32* %1) nounwind
	ret void
}

declare void @g(i32*, i32*)
