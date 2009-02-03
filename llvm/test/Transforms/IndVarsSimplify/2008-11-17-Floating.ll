; RUN: llvm-as < %s | opt -indvars | llvm-dis | grep icmp | count 2
; RUN: llvm-as < %s | opt -indvars | llvm-dis | grep sitofp | count 1
; RUN: llvm-as < %s | opt -indvars | llvm-dis | grep uitofp | count 1

define void @bar() nounwind {
entry:
	br label %bb

bb:		; preds = %bb, %entry
	%x.0.reg2mem.0 = phi double [ 0.000000e+00, %entry ], [ %1, %bb ]		; <double> [#uses=2]
	%0 = tail call i32 @foo(double %x.0.reg2mem.0) nounwind		; <i32> [#uses=0]
	%1 = add double %x.0.reg2mem.0, 1.0e+0		; <double> [#uses=2]
	%2 = fcmp olt double %1, 2147483646.0e+0		; <i1> [#uses=1]
	br i1 %2, label %bb, label %return

return:		; preds = %bb
	ret void
}

define void @bar1() nounwind {
entry:
	br label %bb

bb:		; preds = %bb, %entry
	%x.0.reg2mem.0 = phi double [ 0.000000e+00, %entry ], [ %1, %bb ]		; <double> [#uses=2]
	%0 = tail call i32 @foo(double %x.0.reg2mem.0) nounwind		; <i32> [#uses=0]
	%1 = add double %x.0.reg2mem.0, 1.0e+0		; <double> [#uses=2]
	%2 = fcmp olt double %1, 2147483647.0e+0		; <i1> [#uses=1]
	br i1 %2, label %bb, label %return

return:		; preds = %bb
	ret void
}

declare i32 @foo(double)
