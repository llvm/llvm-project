; RUN: llc < %s -march=x86-64 > %t
; RUN: grep cmov %t | count 2
; RUN: grep jne %t | count 1

; LSR's OptimizeMax function shouldn't try to eliminate this max, because
; it has three operands.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define void @foo(double* nocapture %p, i64 %x, i64 %y) nounwind {
entry:
	%tmp = icmp eq i64 %y, 0		; <i1> [#uses=1]
	%umax = select i1 %tmp, i64 1, i64 %y		; <i64> [#uses=2]
	%tmp8 = icmp ugt i64 %umax, %x		; <i1> [#uses=1]
	%umax9 = select i1 %tmp8, i64 %umax, i64 %x		; <i64> [#uses=1]
	br label %bb4

bb4:		; preds = %bb4, %entry
	%i.07 = phi i64 [ 0, %entry ], [ %2, %bb4 ]		; <i64> [#uses=2]
	%scevgep = getelementptr double* %p, i64 %i.07		; <double*> [#uses=2]
	%0 = load double* %scevgep, align 8		; <double> [#uses=1]
	%1 = fmul double %0, 2.000000e+00		; <double> [#uses=1]
	store double %1, double* %scevgep, align 8
	%2 = add i64 %i.07, 1		; <i64> [#uses=2]
	%exitcond = icmp eq i64 %2, %umax9		; <i1> [#uses=1]
	br i1 %exitcond, label %return, label %bb4

return:		; preds = %bb4
	ret void
}
