; RUN: llc < %s -march=x86
; PR2775

define i32 @func_77(i8 zeroext %p_79) nounwind {
entry:
	%0 = tail call i32 (...) @func_43(i32 1) nounwind		; <i32> [#uses=1]
	%1 = icmp eq i32 %0, 0		; <i1> [#uses=1]
	br i1 %1, label %bb3, label %bb

bb:		; preds = %entry
	br label %bb3

bb3:		; preds = %bb, %entry
	%p_79_addr.0 = phi i8 [ 0, %bb ], [ %p_79, %entry ]		; <i8> [#uses=1]
	%2 = zext i8 %p_79_addr.0 to i32		; <i32> [#uses=2]
	%3 = zext i1 false to i32		; <i32> [#uses=2]
	%4 = tail call i32 (...) @rshift_u_s(i32 1) nounwind		; <i32> [#uses=0]
	%5 = lshr i32 %2, %2		; <i32> [#uses=3]
	%6 = icmp eq i32 0, 0		; <i1> [#uses=1]
	br i1 %6, label %bb6, label %bb9

bb6:		; preds = %bb3
	%7 = ashr i32 %5, %3		; <i32> [#uses=1]
	%8 = icmp eq i32 %7, 0		; <i1> [#uses=1]
	%9 = select i1 %8, i32 %3, i32 0		; <i32> [#uses=1]
	%. = shl i32 %5, %9		; <i32> [#uses=1]
	br label %bb9

bb9:		; preds = %bb6, %bb3
	%.0 = phi i32 [ %., %bb6 ], [ %5, %bb3 ]		; <i32> [#uses=0]
	br i1 false, label %return, label %bb10

bb10:		; preds = %bb9
	ret i32 undef

return:		; preds = %bb9
	ret i32 undef
}

declare i32 @func_43(...)

declare i32 @rshift_u_s(...)
