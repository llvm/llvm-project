; RUN: llc < %s

define void @QRiterate(i32 %p.1, double %tmp.212) nounwind {
entry:
	br i1 false, label %shortcirc_next.1, label %exit.1.critedge

shortcirc_next.1:		; preds = %shortcirc_next.1, %entry
	%tmp.213 = fcmp une double %tmp.212, 0.000000e+00		; <i1> [#uses=1]
	br i1 %tmp.213, label %shortcirc_next.1, label %exit.1

exit.1.critedge:		; preds = %entry
	ret void

exit.1:		; preds = %shortcirc_next.1
	ret void
}
