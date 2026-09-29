; RUN: llc < %s
; verify PR 1700 is still fixed
; ModuleID = 'hh.c'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"

define x86_fp80 @x() {
entry:
	%retval = alloca x86_fp80, align 16		; <ptr> [#uses=2]
	%tmp = alloca x86_fp80, align 16		; <ptr> [#uses=2]
	%d = alloca double, align 8		; <ptr> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store double 0.000000e+00, ptr %d, align 8
	%tmp1 = load double, ptr %d, align 8		; <double> [#uses=1]
	%tmp12 = fpext double %tmp1 to x86_fp80		; <x86_fp80> [#uses=1]
	store x86_fp80 %tmp12, ptr %tmp, align 16
	%tmp3 = load x86_fp80, ptr %tmp, align 16		; <x86_fp80> [#uses=1]
	store x86_fp80 %tmp3, ptr %retval, align 16
	br label %return

return:		; preds = %entry
	%retval4 = load x86_fp80, ptr %retval		; <x86_fp80> [#uses=1]
	ret x86_fp80 %retval4
}

define double @y() {
entry:
	%retval = alloca double, align 8		; <ptr> [#uses=2]
	%tmp = alloca double, align 8		; <ptr> [#uses=2]
	%ld = alloca x86_fp80, align 16		; <ptr> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store x86_fp80 0xK00000000000000000000, ptr %ld, align 16
	%tmp1 = load x86_fp80, ptr %ld, align 16		; <x86_fp80> [#uses=1]
	%tmp12 = fptrunc x86_fp80 %tmp1 to double		; <double> [#uses=1]
	store double %tmp12, ptr %tmp, align 8
	%tmp3 = load double, ptr %tmp, align 8		; <double> [#uses=1]
	store double %tmp3, ptr %retval, align 8
	br label %return

return:		; preds = %entry
	%retval4 = load double, ptr %retval		; <double> [#uses=1]
	ret double %retval4
}
