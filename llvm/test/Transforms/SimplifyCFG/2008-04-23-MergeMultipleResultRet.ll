; RUN: llvm-as < %s | opt -simplifycfg -disable-output
; rdar://5882392
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin9"
	%struct.Py_complex = type { double, double }

define %struct.Py_complex @_Py_c_pow(double %a.0, double %a.1, double %b.0, double %b.1) nounwind  {
entry:
	%tmp7 = fcmp une double %b.0, 0.000000e+00		; <i1> [#uses=1]
	%tmp11 = fcmp une double %b.1, 0.000000e+00		; <i1> [#uses=1]
	%bothcond = or i1 %tmp7, %tmp11		; <i1> [#uses=1]
	br i1 %bothcond, label %bb15, label %bb53

bb15:		; preds = %entry
	%tmp18 = fcmp une double %a.0, 0.000000e+00		; <i1> [#uses=1]
	%tmp24 = fcmp une double %a.1, 0.000000e+00		; <i1> [#uses=1]
	%bothcond1 = or i1 %tmp18, %tmp24		; <i1> [#uses=1]
	br i1 %bothcond1, label %bb29, label %bb27

bb27:		; preds = %bb15
	%tmp28 = call i32* @__error( ) nounwind 		; <i32*> [#uses=1]
	store i32 33, i32* %tmp28, align 4
	ret double undef, double undef

bb29:		; preds = %bb15
	%tmp36 = fcmp une double %b.1, 0.000000e+00		; <i1> [#uses=1]
	br i1 %tmp36, label %bb39, label %bb47

bb39:		; preds = %bb29
	br label %bb47

bb47:		; preds = %bb39, %bb29
	ret double undef, double undef

bb53:		; preds = %entry
	ret double undef, double undef
}

declare i32* @__error()

declare double @pow(double, double) nounwind readonly 

declare double @cos(double) nounwind readonly 
