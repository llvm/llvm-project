; RUN: llvm-as < %s | llc -mtriple=arm-linux-gnueabi | grep powidf2
; PR1287

; ModuleID = '<stdin>'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "arm-linux-gnueabi"

define double @_ZSt3powdi(double %__x, i32 %__i) {
entry:
	%tmp3 = call double @llvm.powi.f64( double 0.000000e+00, i32 0 )		; <double> [#uses=1]
	store double %tmp3, double* null, align 8
	unreachable
}

declare double @llvm.powi.f64(double, i32)

