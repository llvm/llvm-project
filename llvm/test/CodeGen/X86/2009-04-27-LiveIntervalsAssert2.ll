; RUN: llc < %s -mtriple=i386-apple-darwin9
; PR4051

define void @int163(i32 %p_4, i32 %p_5) nounwind {
entry:
	%0 = tail call i32 @foo(i32 1) nounwind		; <i32> [#uses=2]
	%1 = icmp eq i32 %0, 0		; <i1> [#uses=1]
	br i1 %1, label %bb.i, label %bar.exit

bb.i:		; preds = %entry
	%2 = lshr i32 1, %0		; <i32> [#uses=1]
	%3 = icmp eq i32 %2, 0		; <i1> [#uses=1]
	%retval.i = select i1 %3, i32 1, i32 %p_5		; <i32> [#uses=1]
	br label %bar.exit

bar.exit:		; preds = %bb.i, %entry
	%4 = phi i32 [ %retval.i, %bb.i ], [ %p_5, %entry ]		; <i32> [#uses=1]
	%5 = icmp eq i32 %4, 0		; <i1> [#uses=0]
	%6 = tail call i32 @foo(i32 %p_5) nounwind		; <i32> [#uses=0]
	ret void
}

declare i32 @foo(i32)
