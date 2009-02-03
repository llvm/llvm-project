; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 -pre-alloc-split -stats |& \
; RUN:   grep {pre-alloc-split} | grep {Number of intervals split} | grep 1
; XFAIL: *

define void @test(double* %P, i32 %cond) nounwind {
entry:
	%0 = load double* %P, align 8		; <double> [#uses=1]
	%1 = add double %0, 4.000000e+00		; <double> [#uses=2]
	%2 = icmp eq i32 %cond, 0		; <i1> [#uses=1]
	br i1 %2, label %bb1, label %bb

bb:		; preds = %entry
	%3 = add double %1, 4.000000e+00		; <double> [#uses=1]
	br label %bb1

bb1:		; preds = %bb, %entry
	%A.0 = phi double [ %3, %bb ], [ %1, %entry ]		; <double> [#uses=1]
	%4 = mul double %A.0, 4.000000e+00		; <double> [#uses=1]
	%5 = tail call i32 (...)* @bar() nounwind		; <i32> [#uses=0]
	store double %4, double* %P, align 8
	ret void
}

declare i32 @bar(...)
