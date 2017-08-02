; RUN: llc < %s -mtriple=i686--
; PR3486

define i32 @foo(i8 signext %p_26) nounwind {
entry:
	%0 = icmp eq i8 %p_26, 0		; <i1> [#uses=2]
	%or.cond = or i1 false, %0		; <i1> [#uses=2]
	%iftmp.1.0 = zext i1 %or.cond to i16		; <i16> [#uses=1]
	br i1 %0, label %bb.i, label %bar.exit

bb.i:		; preds = %entry
	%1 = zext i1 %or.cond to i32		; <i32> [#uses=1]
	%2 = sdiv i32 %1, 0		; <i32> [#uses=1]
	%3 = trunc i32 %2 to i16		; <i16> [#uses=1]
	br label %bar.exit

bar.exit:		; preds = %bb.i, %entry
	%4 = phi i16 [ %3, %bb.i ], [ %iftmp.1.0, %entry ]		; <i16> [#uses=1]
	%5 = trunc i16 %4 to i8		; <i8> [#uses=1]
	%6 = sext i8 %5 to i32		; <i32> [#uses=1]
	ret i32 %6
}
