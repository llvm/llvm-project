; RUN: llc < %s -mtriple=i386-apple-darwin9
; PR4056

define void @int163(i32 %p_4, i32 %p_5) nounwind {
entry:
	%0 = tail call i32 @bar(i32 1) nounwind		; <i32> [#uses=2]
	%1 = icmp sgt i32 %0, 7		; <i1> [#uses=1]
	br i1 %1, label %foo.exit, label %bb.i

bb.i:		; preds = %entry
	%2 = lshr i32 1, %0		; <i32> [#uses=1]
	%3 = icmp eq i32 %2, 0		; <i1> [#uses=1]
	%4 = zext i1 %3 to i32		; <i32> [#uses=1]
	%.p_5 = shl i32 %p_5, %4		; <i32> [#uses=1]
	br label %foo.exit

foo.exit:		; preds = %bb.i, %entry
	%5 = phi i32 [ %.p_5, %bb.i ], [ %p_5, %entry ]		; <i32> [#uses=1]
	%6 = icmp eq i32 %5, 0		; <i1> [#uses=0]
	%7 = tail call i32 @bar(i32 %p_5) nounwind		; <i32> [#uses=0]
	ret void
}

declare i32 @bar(i32)
