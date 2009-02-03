; RUN: llvm-as < %s | opt -simplify-libcalls-halfpowr | llvm-dis | %prcontext {mul float} 1 | grep {mul float} | count 8

define float @__half_powrf4(float %f, float %g) nounwind readnone {
entry:
	%0 = fcmp olt float %f, 2.000000e+00		; <i1> [#uses=1]
	br i1 %0, label %bb, label %bb1

bb:		; preds = %entry
	%1 = fdiv float %f, 3.000000e+00		; <float> [#uses=1]
	br label %bb1

bb1:		; preds = %bb, %entry
	%f_addr.0 = phi float [ %1, %bb ], [ %f, %entry ]		; <float> [#uses=1]
	%2 = mul float %f_addr.0, %g		; <float> [#uses=1]
	ret float %2
}

define void @foo(float* %p) nounwind {
entry:
	%0 = load float* %p, align 4		; <float> [#uses=1]
	%1 = getelementptr float* %p, i32 1		; <float*> [#uses=1]
	%2 = load float* %1, align 4		; <float> [#uses=1]
	%3 = getelementptr float* %p, i32 2		; <float*> [#uses=1]
	%4 = load float* %3, align 4		; <float> [#uses=1]
	%5 = getelementptr float* %p, i32 3		; <float*> [#uses=1]
	%6 = load float* %5, align 4		; <float> [#uses=1]
	%7 = getelementptr float* %p, i32 4		; <float*> [#uses=1]
	%8 = load float* %7, align 4		; <float> [#uses=1]
	%9 = getelementptr float* %p, i32 5		; <float*> [#uses=1]
	%10 = load float* %9, align 4		; <float> [#uses=1]
	%11 = tail call float @__half_powrf4(float %0, float %6) nounwind		; <float> [#uses=1]
	%12 = tail call float @__half_powrf4(float %2, float %8) nounwind		; <float> [#uses=1]
	%13 = tail call float @__half_powrf4(float %4, float %10) nounwind		; <float> [#uses=1]
	%14 = getelementptr float* %p, i32 6		; <float*> [#uses=1]
	store float %11, float* %14, align 4
	%15 = getelementptr float* %p, i32 7		; <float*> [#uses=1]
	store float %12, float* %15, align 4
	%16 = getelementptr float* %p, i32 8		; <float*> [#uses=1]
	store float %13, float* %16, align 4
	ret void
}
