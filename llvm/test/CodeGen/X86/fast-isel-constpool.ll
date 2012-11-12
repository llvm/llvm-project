; RUN: llc < %s -fast-isel | grep "LCPI0_0(%rip)"
; Make sure fast isel uses rip-relative addressing when required.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin9.0"

define i32 @f0(double %x) nounwind {
entry:
	%retval = alloca i32		; <i32*> [#uses=2]
	%x.addr = alloca double		; <double*> [#uses=2]
	store double %x, double* %x.addr
	%tmp = load double* %x.addr		; <double> [#uses=1]
	%cmp = fcmp olt double %tmp, 8.500000e-01		; <i1> [#uses=1]
	%conv = zext i1 %cmp to i32		; <i32> [#uses=1]
	store i32 %conv, i32* %retval
	%0 = load i32* %retval		; <i32> [#uses=1]
	ret i32 %0
}
