; RUN: llvm-as < %s | llvm-dis > %t
; RUN: llvm-as < %t | llvm-dis > %t2
; RUN: diff %t %t2
; ModuleID = '<stdin>'
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "powerpc-apple-darwin8"
@ld = external global ppc_fp128		; <ppc_fp128*> [#uses=1]
@d = global double 4.050000e+00, align 8		; <double*> [#uses=1]
@f = global float 0x4010333340000000		; <float*> [#uses=1]

define i32 @foo() {
entry:
	%retval = alloca i32, align 4		; <i32*> [#uses=1]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%tmp = load float* @f		; <float> [#uses=1]
	%tmp1 = fpext float %tmp to double		; <double> [#uses=1]
	%tmp2 = load double* @d		; <double> [#uses=1]
	%tmp3 = mul double %tmp1, %tmp2		; <double> [#uses=1]
	%tmp4 = fpext double %tmp3 to ppc_fp128		; <ppc_fp128> [#uses=1]
	store ppc_fp128 %tmp4, ppc_fp128* @ld
	br label %return

return:		; preds = %entry
	%retval4 = load i32* %retval		; <i32> [#uses=1]
	ret i32 %retval4
}
