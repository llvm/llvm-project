; RUN: llc < %s -march=arm | grep "bl.*__ltdf" | count 1
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"

; Without CSE of libcalls, there are two calls in the output instead of one.

define double @u_f_nonbon(double %lambda) nounwind {
entry:
	%tmp19.i.i = load double* null, align 4		; <double> [#uses=2]
	%tmp6.i = fcmp olt double %tmp19.i.i, 1.000000e+00		; <i1> [#uses=1]
	%dielectric.0.i = select i1 %tmp6.i, double 1.000000e+00, double %tmp19.i.i		; <double> [#uses=1]
	%tmp10.i4 = fdiv double 0x4074C2D71F36262D, %dielectric.0.i		; <double> [#uses=1]
	br i1 false, label %bb28.i, label %bb508.i

bb28.i:		; preds = %bb28.i, %entry
	br i1 false, label %bb502.loopexit.i, label %bb28.i

bb.nph53.i:		; preds = %bb502.loopexit.i
	%tmp354.i = fsub double -0.000000e+00, %tmp10.i4		; <double> [#uses=0]
	br label %bb244.i

bb244.i:		; preds = %bb244.i, %bb.nph53.i
	br label %bb244.i

bb502.loopexit.i:		; preds = %bb28.i
	br i1 false, label %bb.nph53.i, label %bb508.i

bb508.i:		; preds = %bb502.loopexit.i, %entry
	ret double %tmp10.i4
}
