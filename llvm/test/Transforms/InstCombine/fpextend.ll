; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep fpext
@X = external global float 
@Y = external global float

define void @test() nounwind  {
entry:
	%tmp = load float* @X, align 4		; <float> [#uses=1]
	%tmp1 = fpext float %tmp to double		; <double> [#uses=1]
	%tmp3 = add double %tmp1, 0.000000e+00		; <double> [#uses=1]
	%tmp34 = fptrunc double %tmp3 to float		; <float> [#uses=1]
	store float %tmp34, float* @X, align 4
	ret void
}

define void @test3() nounwind  {
entry:
	%tmp = load float* @X, align 4		; <float> [#uses=1]
	%tmp1 = fpext float %tmp to double		; <double> [#uses=1]
	%tmp2 = load float* @Y, align 4		; <float> [#uses=1]
	%tmp23 = fpext float %tmp2 to double		; <double> [#uses=1]
	%tmp5 = fdiv double %tmp1, %tmp23		; <double> [#uses=1]
	%tmp56 = fptrunc double %tmp5 to float		; <float> [#uses=1]
	store float %tmp56, float* @X, align 4
	ret void
}

define void @test4() nounwind  {
entry:
	%tmp = load float* @X, align 4		; <float> [#uses=1]
	%tmp1 = fpext float %tmp to double		; <double> [#uses=1]
	%tmp2 = sub double -0.000000e+00, %tmp1		; <double> [#uses=1]
	%tmp34 = fptrunc double %tmp2 to float		; <float> [#uses=1]
	store float %tmp34, float* @X, align 4
	ret void
}

