; RUN: llvm-as < %s | opt -simplifycfg -instcombine | llvm-dis | grep 0x3FB99999A0000000 | count 2
; RUN: llvm-as < %s | opt -simplifycfg -instcombine | llvm-dis | grep 0xBFB99999A0000000 | count 2
; check constant folding for 'frem'.  PR 3316.

; ModuleID = 'tt.c'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.6"

define float @test1() nounwind {
entry:
	%retval = alloca float		; <float*> [#uses=2]
	%0 = alloca float		; <float*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%1 = frem double 1.000000e-01, 1.000000e+00	; <double> [#uses=1]
	%2 = fptrunc double %1 to float		; <float> [#uses=1]
	store float %2, float* %0, align 4
	%3 = load float* %0, align 4		; <float> [#uses=1]
	store float %3, float* %retval, align 4
	br label %return

return:		; preds = %entry
	%retval1 = load float* %retval		; <float> [#uses=1]
	ret float %retval1
}

define float @test2() nounwind {
entry:
	%retval = alloca float		; <float*> [#uses=2]
	%0 = alloca float		; <float*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%1 = frem double -1.000000e-01, 1.000000e+00	; <double> [#uses=1]
	%2 = fptrunc double %1 to float		; <float> [#uses=1]
	store float %2, float* %0, align 4
	%3 = load float* %0, align 4		; <float> [#uses=1]
	store float %3, float* %retval, align 4
	br label %return

return:		; preds = %entry
	%retval1 = load float* %retval		; <float> [#uses=1]
	ret float %retval1
}

define float @test3() nounwind {
entry:
	%retval = alloca float		; <float*> [#uses=2]
	%0 = alloca float		; <float*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%1 = frem double 1.000000e-01, -1.000000e+00	; <double> [#uses=1]
	%2 = fptrunc double %1 to float		; <float> [#uses=1]
	store float %2, float* %0, align 4
	%3 = load float* %0, align 4		; <float> [#uses=1]
	store float %3, float* %retval, align 4
	br label %return

return:		; preds = %entry
	%retval1 = load float* %retval		; <float> [#uses=1]
	ret float %retval1
}

define float @test4() nounwind {
entry:
	%retval = alloca float		; <float*> [#uses=2]
	%0 = alloca float		; <float*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%1 = frem double -1.000000e-01, -1.000000e+00	; <double> [#uses=1]
	%2 = fptrunc double %1 to float		; <float> [#uses=1]
	store float %2, float* %0, align 4
	%3 = load float* %0, align 4		; <float> [#uses=1]
	store float %3, float* %retval, align 4
	br label %return

return:		; preds = %entry
	%retval1 = load float* %retval		; <float> [#uses=1]
	ret float %retval1
}
