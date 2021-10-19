; RUN: llc < %s -mtriple=i686--
; PR2783

@g_15 = external dso_local global i16		; <i16*> [#uses=2]

define i32 @func_3(i32 %p_5) nounwind {
entry:
	%0 = srem i32 1, 0		; <i32> [#uses=2]
	%1 = load i16, i16* @g_15, align 2		; <i16> [#uses=1]
	%2 = zext i16 %1 to i32		; <i32> [#uses=1]
	%3 = and i32 %2, 1		; <i32> [#uses=1]
	%4 = tail call i32 (...) @rshift_u_s( i32 1 ) nounwind		; <i32> [#uses=1]
	%5 = icmp slt i32 %4, 2		; <i1> [#uses=1]
	%6 = zext i1 %5 to i32		; <i32> [#uses=1]
	%7 = icmp sge i32 %3, %6		; <i1> [#uses=1]
	%8 = zext i1 %7 to i32		; <i32> [#uses=1]
	%9 = load i16, i16* @g_15, align 2		; <i16> [#uses=1]
	%10 = icmp eq i16 %9, 0		; <i1> [#uses=1]
	%11 = zext i1 %10 to i32		; <i32> [#uses=1]
	%12 = tail call i32 (...) @func_20( i32 1 ) nounwind		; <i32> [#uses=1]
	%13 = icmp sge i32 %11, %12		; <i1> [#uses=1]
	%14 = zext i1 %13 to i32		; <i32> [#uses=1]
	%15 = sub i32 %8, %14		; <i32> [#uses=1]
	%16 = icmp ult i32 %15, 2		; <i1> [#uses=1]
	%17 = zext i1 %16 to i32		; <i32> [#uses=1]
	%18 = icmp ugt i32 %0, 3		; <i1> [#uses=1]
	%or.cond = or i1 false, %18		; <i1> [#uses=1]
	%19 = select i1 %or.cond, i32 0, i32 %0		; <i32> [#uses=1]
	%.0 = lshr i32 %17, %19		; <i32> [#uses=1]
	%20 = tail call i32 (...) @func_7( i32 %.0 ) nounwind		; <i32> [#uses=0]
	ret i32 undef
}

declare i32 @rshift_u_s(...)

declare i32 @func_20(...)

declare i32 @func_7(...)
