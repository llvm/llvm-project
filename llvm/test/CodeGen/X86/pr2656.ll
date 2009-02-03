; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep {xorps.\*sp} | count 1
; PR2656

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin9.4.0"
	%struct.anon = type <{ float, float }>
@.str = internal constant [17 x i8] c"pt: %.0f, %.0f\0A\00\00"		; <[17 x i8]*> [#uses=1]

define void @foo(%struct.anon* byval %p) nounwind {
entry:
	%tmp = getelementptr %struct.anon* %p, i32 0, i32 0		; <float*> [#uses=1]
	%tmp1 = load float* %tmp		; <float> [#uses=1]
	%tmp2 = getelementptr %struct.anon* %p, i32 0, i32 1		; <float*> [#uses=1]
	%tmp3 = load float* %tmp2		; <float> [#uses=1]
	%neg = sub float -0.000000e+00, %tmp1		; <float> [#uses=1]
	%conv = fpext float %neg to double		; <double> [#uses=1]
	%neg4 = sub float -0.000000e+00, %tmp3		; <float> [#uses=1]
	%conv5 = fpext float %neg4 to double		; <double> [#uses=1]
	%call = call i32 (...)* @printf( i8* getelementptr ([17 x i8]* @.str, i32 0, i32 0), double %conv, double %conv5 )		; <i32> [#uses=0]
	ret void
}

declare i32 @printf(...)
