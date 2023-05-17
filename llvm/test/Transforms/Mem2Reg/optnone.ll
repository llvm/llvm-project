; RUN: opt < %s -passes=mem2reg -S | FileCheck %s

; This function is optnone, so the allocas should not be eliminated.

; CHECK-LABEL: @testfunc
; CHECK: alloca
; CHECK: alloca
define double @testfunc(i32 %i, double %j) optnone noinline {
	%I = alloca i32		; <ptr> [#uses=4]
	%J = alloca double		; <ptr> [#uses=2]
	store i32 %i, ptr %I
	store double %j, ptr %J
	%t1 = load i32, ptr %I		; <i32> [#uses=1]
	%t2 = add i32 %t1, 1		; <i32> [#uses=1]
	store i32 %t2, ptr %I
	%t3 = load i32, ptr %I		; <i32> [#uses=1]
	%t4 = sitofp i32 %t3 to double		; <double> [#uses=1]
	%t5 = load double, ptr %J		; <double> [#uses=1]
	%t6 = fmul double %t4, %t5		; <double> [#uses=1]
	ret double %t6
}
