; RUN: llvm-as < %s | opt -indvars | llvm-dis | grep icmp | count 4
define void @bar() nounwind {
entry:
	br label %bb

bb:		; preds = %bb, %entry
	%x.0.reg2mem.0 = phi double [ 0.000000e+00, %entry ], [ %1, %bb ]		; <double> [#uses=2]
	%0 = tail call i32 @foo(double %x.0.reg2mem.0) nounwind		; <i32> [#uses=0]
	%1 = add double %x.0.reg2mem.0, 1.000000e+00		; <double> [#uses=2]
	%2 = fcmp olt double %1, 1.000000e+04		; <i1> [#uses=1]
	br i1 %2, label %bb, label %return

return:		; preds = %bb
	ret void
}

declare i32 @foo(double)

define void @bar2() nounwind {
entry:
	br label %bb

bb:		; preds = %bb, %entry
	%x.0.reg2mem.0 = phi double [ -10.000000e+00, %entry ], [ %1, %bb ]		; <double> [#uses=2]
	%0 = tail call i32 @foo(double %x.0.reg2mem.0) nounwind		; <i32> [#uses=0]
	%1 = add double %x.0.reg2mem.0, 2.000000e+00		; <double> [#uses=2]
	%2 = fcmp olt double %1, -1.000000e+00		; <i1> [#uses=1]
	br i1 %2, label %bb, label %return

return:		; preds = %bb
	ret void
}


define void @bar3() nounwind {
entry:
	br label %bb

bb:		; preds = %bb, %entry
	%x.0.reg2mem.0 = phi double [ 0.000000e+00, %entry ], [ %1, %bb ]		; <double> [#uses=2]
	%0 = tail call i32 @foo(double %x.0.reg2mem.0) nounwind		; <i32> [#uses=0]
	%1 = add double %x.0.reg2mem.0, 1.000000e+00		; <double> [#uses=2]
	%2 = fcmp olt double %1, -1.000000e+00		; <i1> [#uses=1]
	br i1 %2, label %bb, label %return

return:		; preds = %bb
	ret void
}

define void @bar4() nounwind {
entry:
	br label %bb

bb:		; preds = %bb, %entry
	%x.0.reg2mem.0 = phi double [ 40.000000e+00, %entry ], [ %1, %bb ]		; <double> [#uses=2]
	%0 = tail call i32 @foo(double %x.0.reg2mem.0) nounwind		; <i32> [#uses=0]
	%1 = add double %x.0.reg2mem.0, -1.000000e+00		; <double> [#uses=2]
	%2 = fcmp olt double %1, 1.000000e+00		; <i1> [#uses=1]
	br i1 %2, label %bb, label %return

return:		; preds = %bb
	ret void
}


