; RUN: opt < %s -indvars -S | grep add | count 1

; Indvars should be able to compute the exit value of this loop
; without any additional arithmetic. The only add needed should
; be the canonical IV increment.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define void @foo(double* %p, i32 %n) nounwind {
entry:
	%0 = icmp sgt i32 %n, 0		; <i1> [#uses=1]
	br i1 %0, label %bb.nph, label %return

bb.nph:		; preds = %entry
	br label %bb2

bb2:		; preds = %bb3, %bb.nph
	%i.01 = phi i32 [ %7, %bb3 ], [ 0, %bb.nph ]		; <i32> [#uses=3]
	%1 = sext i32 %i.01 to i64		; <i64> [#uses=1]
	%2 = getelementptr double* %p, i64 %1		; <double*> [#uses=1]
	%3 = load double* %2, align 8		; <double> [#uses=1]
	%4 = fmul double %3, 1.100000e+00		; <double> [#uses=1]
	%5 = sext i32 %i.01 to i64		; <i64> [#uses=1]
	%6 = getelementptr double* %p, i64 %5		; <double*> [#uses=1]
	store double %4, double* %6, align 8
	%7 = add i32 %i.01, 1		; <i32> [#uses=2]
	br label %bb3

bb3:		; preds = %bb2
	%8 = icmp slt i32 %7, %n		; <i1> [#uses=1]
	br i1 %8, label %bb2, label %bb3.return_crit_edge

bb3.return_crit_edge:		; preds = %bb3
	br label %return

return:		; preds = %bb3.return_crit_edge, %entry
	ret void
}
