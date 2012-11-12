; RUN: llc < %s -march=x86-64 | not grep cmov

; LSR should be able to eliminate both smax and umax expressions
; in loop trip counts.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define void @fs(double* nocapture %p, i64 %n) nounwind {
entry:
	%tmp = icmp slt i64 %n, 1		; <i1> [#uses=1]
	%smax = select i1 %tmp, i64 1, i64 %n		; <i64> [#uses=1]
	br label %bb

bb:		; preds = %bb, %entry
	%i.0 = phi i64 [ 0, %entry ], [ %0, %bb ]		; <i64> [#uses=2]
	%scevgep = getelementptr double* %p, i64 %i.0		; <double*> [#uses=1]
	store double 0.000000e+00, double* %scevgep, align 8
	%0 = add i64 %i.0, 1		; <i64> [#uses=2]
	%exitcond = icmp eq i64 %0, %smax		; <i1> [#uses=1]
	br i1 %exitcond, label %return, label %bb

return:		; preds = %bb
	ret void
}

define void @bs(double* nocapture %p, i64 %n) nounwind {
entry:
	%tmp = icmp sge i64 %n, 1		; <i1> [#uses=1]
	%smax = select i1 %tmp, i64 %n, i64 1		; <i64> [#uses=1]
	br label %bb

bb:		; preds = %bb, %entry
	%i.0 = phi i64 [ 0, %entry ], [ %0, %bb ]		; <i64> [#uses=2]
	%scevgep = getelementptr double* %p, i64 %i.0		; <double*> [#uses=1]
	store double 0.000000e+00, double* %scevgep, align 8
	%0 = add i64 %i.0, 1		; <i64> [#uses=2]
	%exitcond = icmp eq i64 %0, %smax		; <i1> [#uses=1]
	br i1 %exitcond, label %return, label %bb

return:		; preds = %bb
	ret void
}

define void @fu(double* nocapture %p, i64 %n) nounwind {
entry:
	%tmp = icmp eq i64 %n, 0		; <i1> [#uses=1]
	%umax = select i1 %tmp, i64 1, i64 %n		; <i64> [#uses=1]
	br label %bb

bb:		; preds = %bb, %entry
	%i.0 = phi i64 [ 0, %entry ], [ %0, %bb ]		; <i64> [#uses=2]
	%scevgep = getelementptr double* %p, i64 %i.0		; <double*> [#uses=1]
	store double 0.000000e+00, double* %scevgep, align 8
	%0 = add i64 %i.0, 1		; <i64> [#uses=2]
	%exitcond = icmp eq i64 %0, %umax		; <i1> [#uses=1]
	br i1 %exitcond, label %return, label %bb

return:		; preds = %bb
	ret void
}

define void @bu(double* nocapture %p, i64 %n) nounwind {
entry:
	%tmp = icmp ne i64 %n, 0		; <i1> [#uses=1]
	%umax = select i1 %tmp, i64 %n, i64 1		; <i64> [#uses=1]
	br label %bb

bb:		; preds = %bb, %entry
	%i.0 = phi i64 [ 0, %entry ], [ %0, %bb ]		; <i64> [#uses=2]
	%scevgep = getelementptr double* %p, i64 %i.0		; <double*> [#uses=1]
	store double 0.000000e+00, double* %scevgep, align 8
	%0 = add i64 %i.0, 1		; <i64> [#uses=2]
	%exitcond = icmp eq i64 %0, %umax		; <i1> [#uses=1]
	br i1 %exitcond, label %return, label %bb

return:		; preds = %bb
	ret void
}
