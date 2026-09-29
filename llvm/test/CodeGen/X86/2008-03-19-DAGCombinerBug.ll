; RUN: llc < %s -mtriple=i686--

define i32 @t() nounwind  {
entry:
	%tmp54 = add i32 0, 1		; <i32> [#uses=1]
	br i1 false, label %bb71, label %bb77
bb71:		; preds = %entry
	%tmp74 = shl i32 %tmp54, 1		; <i32> [#uses=1]
	%tmp76 = ashr i32 %tmp74, 3		; <i32> [#uses=1]
	br label %bb77
bb77:		; preds = %bb71, %entry
	%payLoadSize.0 = phi i32 [ %tmp76, %bb71 ], [ 0, %entry ]		; <i32> [#uses=0]
	unreachable
}
