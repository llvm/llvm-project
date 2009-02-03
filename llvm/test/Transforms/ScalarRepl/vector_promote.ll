; RUN: llvm-as < %s | opt -scalarrepl | llvm-dis | not grep alloca
; RUN: llvm-as < %s | opt -scalarrepl | llvm-dis | grep {load <4 x float>}

define void @test(<4 x float>* %F, float %f) {
entry:
	%G = alloca <4 x float>, align 16		; <<4 x float>*> [#uses=3]
	%tmp = load <4 x float>* %F		; <<4 x float>> [#uses=2]
	%tmp3 = add <4 x float> %tmp, %tmp		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp3, <4 x float>* %G
	%G.upgrd.1 = getelementptr <4 x float>* %G, i32 0, i32 0		; <float*> [#uses=1]
	store float %f, float* %G.upgrd.1
	%tmp4 = load <4 x float>* %G		; <<4 x float>> [#uses=2]
	%tmp6 = add <4 x float> %tmp4, %tmp4		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp6, <4 x float>* %F
	ret void
}

define void @test2(<4 x float>* %F, float %f) {
entry:
	%G = alloca <4 x float>, align 16		; <<4 x float>*> [#uses=3]
	%tmp = load <4 x float>* %F		; <<4 x float>> [#uses=2]
	%tmp3 = add <4 x float> %tmp, %tmp		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp3, <4 x float>* %G
	%tmp.upgrd.2 = getelementptr <4 x float>* %G, i32 0, i32 2		; <float*> [#uses=1]
	store float %f, float* %tmp.upgrd.2
	%tmp4 = load <4 x float>* %G		; <<4 x float>> [#uses=2]
	%tmp6 = add <4 x float> %tmp4, %tmp4		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp6, <4 x float>* %F
	ret void
}

define void @test3(<4 x float>* %F, float* %f) {
entry:
	%G = alloca <4 x float>, align 16		; <<4 x float>*> [#uses=2]
	%tmp = load <4 x float>* %F		; <<4 x float>> [#uses=2]
	%tmp3 = add <4 x float> %tmp, %tmp		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp3, <4 x float>* %G
	%tmp.upgrd.3 = getelementptr <4 x float>* %G, i32 0, i32 2		; <float*> [#uses=1]
	%tmp.upgrd.4 = load float* %tmp.upgrd.3		; <float> [#uses=1]
	store float %tmp.upgrd.4, float* %f
	ret void
}

define void @test4(<4 x float>* %F, float* %f) {
entry:
	%G = alloca <4 x float>, align 16		; <<4 x float>*> [#uses=2]
	%tmp = load <4 x float>* %F		; <<4 x float>> [#uses=2]
	%tmp3 = add <4 x float> %tmp, %tmp		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp3, <4 x float>* %G
	%G.upgrd.5 = getelementptr <4 x float>* %G, i32 0, i32 0		; <float*> [#uses=1]
	%tmp.upgrd.6 = load float* %G.upgrd.5		; <float> [#uses=1]
	store float %tmp.upgrd.6, float* %f
	ret void
}

define i32 @test5(float %X) {  ;; should turn into bitcast.
	%X_addr = alloca [4 x float]
        %X1 = getelementptr [4 x float]* %X_addr, i32 0, i32 2
	store float %X, float* %X1
	%a = bitcast float* %X1 to i32*
	%tmp = load i32* %a
	ret i32 %tmp
}

