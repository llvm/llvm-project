; RUN: llvm-as < %s | llvm-dis > %t
; RUN: llvm-as < %t | llvm-dis > %t2
; RUN: diff %t %t2
; ModuleID = '<stdin>'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i686-apple-darwin8"
@ld = external global x86_fp80		; <ptr> [#uses=1]
@d = global double 4.050000e+00, align 8		; <ptr> [#uses=1]
@f = global float 0x4010333340000000		; <ptr> [#uses=1]

define i32 @foo() {
entry:
	%retval = alloca i32, align 4		; <ptr> [#uses=1]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%tmp = load float, ptr @f		; <float> [#uses=1]
	%tmp1 = fpext float %tmp to double		; <double> [#uses=1]
	%tmp2 = load double, ptr @d		; <double> [#uses=1]
	%tmp3 = fmul double %tmp1, %tmp2		; <double> [#uses=1]
	%tmp4 = fpext double %tmp3 to x86_fp80		; <x86_fp80> [#uses=1]
	store x86_fp80 %tmp4, ptr @ld
	br label %return

return:		; preds = %entry
	%retval4 = load i32, ptr %retval		; <i32> [#uses=1]
	ret i32 %retval4
}
