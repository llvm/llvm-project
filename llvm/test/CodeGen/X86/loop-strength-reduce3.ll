; RUN: llvm-as < %s | llc -march=x86 | grep cmp | grep 240
; RUN: llvm-as < %s | llc -march=x86 | grep inc | count 1

define i32 @foo(i32 %A, i32 %B, i32 %C, i32 %D) {
entry:
	%tmp2955 = icmp sgt i32 %C, 0		; <i1> [#uses=1]
	br i1 %tmp2955, label %bb26.outer.us, label %bb40.split

bb26.outer.us:		; preds = %bb26.bb32_crit_edge.us, %entry
	%i.044.0.ph.us = phi i32 [ 0, %entry ], [ %indvar.next57, %bb26.bb32_crit_edge.us ]		; <i32> [#uses=2]
	%k.1.ph.us = phi i32 [ 0, %entry ], [ %k.0.us, %bb26.bb32_crit_edge.us ]		; <i32> [#uses=1]
	%tmp3.us = mul i32 %i.044.0.ph.us, 6		; <i32> [#uses=1]
	br label %bb1.us

bb1.us:		; preds = %bb1.us, %bb26.outer.us
	%j.053.us = phi i32 [ 0, %bb26.outer.us ], [ %tmp25.us, %bb1.us ]		; <i32> [#uses=2]
	%k.154.us = phi i32 [ %k.1.ph.us, %bb26.outer.us ], [ %k.0.us, %bb1.us ]		; <i32> [#uses=1]
	%tmp5.us = add i32 %tmp3.us, %j.053.us		; <i32> [#uses=1]
	%tmp7.us = shl i32 %D, %tmp5.us		; <i32> [#uses=2]
	%tmp9.us = icmp eq i32 %tmp7.us, %B		; <i1> [#uses=1]
	%tmp910.us = zext i1 %tmp9.us to i32		; <i32> [#uses=1]
	%tmp12.us = and i32 %tmp7.us, %A		; <i32> [#uses=1]
	%tmp19.us = and i32 %tmp12.us, %tmp910.us		; <i32> [#uses=1]
	%k.0.us = add i32 %tmp19.us, %k.154.us		; <i32> [#uses=3]
	%tmp25.us = add i32 %j.053.us, 1		; <i32> [#uses=2]
	%tmp29.us = icmp slt i32 %tmp25.us, %C		; <i1> [#uses=1]
	br i1 %tmp29.us, label %bb1.us, label %bb26.bb32_crit_edge.us

bb26.bb32_crit_edge.us:		; preds = %bb1.us
	%indvar.next57 = add i32 %i.044.0.ph.us, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next57, 40		; <i1> [#uses=1]
	br i1 %exitcond, label %bb40.split, label %bb26.outer.us

bb40.split:		; preds = %bb26.bb32_crit_edge.us, %entry
	%k.1.lcssa.lcssa.us-lcssa = phi i32 [ %k.0.us, %bb26.bb32_crit_edge.us ], [ 0, %entry ]		; <i32> [#uses=1]
	ret i32 %k.1.lcssa.lcssa.us-lcssa
}
