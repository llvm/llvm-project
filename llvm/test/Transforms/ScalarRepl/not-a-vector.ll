; RUN: opt < %s -scalarrepl -S | not grep alloca
; RUN: opt < %s -scalarrepl -S | not grep "7 x double"
; RUN: opt < %s -scalarrepl -instcombine -S | grep "ret double %B"
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"

define double @test(double %A, double %B) {
	%ARR = alloca [7 x i64]
	%C = bitcast [7 x i64]* %ARR to double*
	store double %A, double* %C

	%D = getelementptr [7 x i64]* %ARR, i32 0, i32 4
	%E = bitcast i64* %D to double*
	store double %B, double* %E

	%F = getelementptr double* %C, i32 4
	%G = load double* %F
	ret double %G
}


