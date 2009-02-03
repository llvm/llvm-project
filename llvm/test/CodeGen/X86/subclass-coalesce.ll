; RUN: llvm-as < %s | llc -march=x86 -join-cross-class-copies -stats |& grep {Number of cross class joins performed}

@mem.6 = external global i64		; <i64*> [#uses=1]

define i64 @attachFunc() nounwind  {
entry:
	%tmp64.i = add i64 0, 72		; <i64> [#uses=1]
	%tmp68.i = load i64* @mem.6, align 8		; <i64> [#uses=1]
	%tmp70.i = icmp sgt i64 %tmp64.i, %tmp68.i		; <i1> [#uses=1]
	br i1 %tmp70.i, label %bb73.i, label %bb116

bb73.i:		; preds = %entry
	br label %bb116

bb116:		; preds = %bb73.i, %entry
	ret i64 %tmp68.i
}
