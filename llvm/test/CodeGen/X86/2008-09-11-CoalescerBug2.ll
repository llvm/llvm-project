; RUN: llc < %s -march=x86
; PR2748

@g_73 = external global i32		; <i32*> [#uses=1]
@g_5 = external global i32		; <i32*> [#uses=1]

define i32 @func_44(i16 signext %p_46) nounwind {
entry:
	%0 = load i32* @g_5, align 4		; <i32> [#uses=1]
	%1 = ashr i32 %0, 1		; <i32> [#uses=1]
	%2 = icmp sgt i32 %1, 1		; <i1> [#uses=1]
	%3 = zext i1 %2 to i32		; <i32> [#uses=1]
	%4 = load i32* @g_73, align 4		; <i32> [#uses=1]
	%5 = zext i16 %p_46 to i64		; <i64> [#uses=1]
	%6 = sub i64 0, %5		; <i64> [#uses=1]
	%7 = trunc i64 %6 to i8		; <i8> [#uses=2]
	%8 = trunc i32 %4 to i8		; <i8> [#uses=2]
	%9 = icmp eq i8 %8, 0		; <i1> [#uses=1]
	br i1 %9, label %bb11, label %bb12

bb11:		; preds = %entry
	%10 = urem i8 %7, %8		; <i8> [#uses=1]
	br label %bb12

bb12:		; preds = %bb11, %entry
	%.014.in = phi i8 [ %10, %bb11 ], [ %7, %entry ]		; <i8> [#uses=1]
	%11 = icmp ne i8 %.014.in, 0		; <i1> [#uses=1]
	%12 = zext i1 %11 to i32		; <i32> [#uses=1]
	%13 = tail call i32 (...)* @func_48( i32 %12, i32 %3, i32 0 ) nounwind		; <i32> [#uses=0]
	ret i32 undef
}

declare i32 @func_48(...)
