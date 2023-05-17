; RUN: opt < %s -passes='loop-mssa(licm),loop-unroll' -disable-output

@resonant = external global i32		; <ptr> [#uses=2]

define void @weightadj() {
entry:
	br label %bb

bb:		; preds = %bb158, %entry
	store i32 0, ptr @resonant, align 4
	br i1 false, label %g.exit, label %bb158

g.exit:		; preds = %bb68, %bb
	br i1 false, label %bb68, label %cond_true

cond_true:		; preds = %g.exit
	store i32 1, ptr @resonant, align 4
	br label %bb68

bb68:		; preds = %cond_true, %g.exit
	%tmp71 = icmp slt i32 0, 0		; <i1> [#uses=1]
	br i1 %tmp71, label %g.exit, label %bb158

bb158:		; preds = %bb68, %bb
	br i1 false, label %bb, label %return

return:		; preds = %bb158
	ret void
}
