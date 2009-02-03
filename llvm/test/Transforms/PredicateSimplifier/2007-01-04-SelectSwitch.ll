; RUN: llvm-as < %s | opt -predsimplify -disable-output

define void @ercMarkCurrMBConcealed(i32 %comp) {
entry:
	%tmp5 = icmp slt i32 %comp, 0		; <i1> [#uses=2]
	%comp_addr.0 = select i1 %tmp5, i32 0, i32 %comp		; <i32> [#uses=1]
	switch i32 %comp_addr.0, label %return [
		 i32 0, label %bb
	]
bb:		; preds = %entry
	br i1 %tmp5, label %bb87.bb97_crit_edge.critedge, label %return
bb87.bb97_crit_edge.critedge:		; preds = %bb
	ret void
return:		; preds = %bb, %entry
	ret void
}

