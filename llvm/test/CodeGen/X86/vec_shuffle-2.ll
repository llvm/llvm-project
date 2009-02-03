; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 -o %t -f
; RUN: grep pshufhw %t | count 1
; RUN: grep pshuflw %t | count 1
; RUN: grep movhps  %t | count 1

define void @test1(<2 x i64>* %res, <2 x i64>* %A) nounwind {
	%tmp = load <2 x i64>* %A		; <<2 x i64>> [#uses=1]
	%tmp.upgrd.1 = bitcast <2 x i64> %tmp to <8 x i16>		; <<8 x i16>> [#uses=8]
	%tmp0 = extractelement <8 x i16> %tmp.upgrd.1, i32 0		; <i16> [#uses=1]
	%tmp1 = extractelement <8 x i16> %tmp.upgrd.1, i32 1		; <i16> [#uses=1]
	%tmp2 = extractelement <8 x i16> %tmp.upgrd.1, i32 2		; <i16> [#uses=1]
	%tmp3 = extractelement <8 x i16> %tmp.upgrd.1, i32 3		; <i16> [#uses=1]
	%tmp4 = extractelement <8 x i16> %tmp.upgrd.1, i32 4		; <i16> [#uses=1]
	%tmp5 = extractelement <8 x i16> %tmp.upgrd.1, i32 5		; <i16> [#uses=1]
	%tmp6 = extractelement <8 x i16> %tmp.upgrd.1, i32 6		; <i16> [#uses=1]
	%tmp7 = extractelement <8 x i16> %tmp.upgrd.1, i32 7		; <i16> [#uses=1]
	%tmp8 = insertelement <8 x i16> undef, i16 %tmp2, i32 0		; <<8 x i16>> [#uses=1]
	%tmp9 = insertelement <8 x i16> %tmp8, i16 %tmp1, i32 1		; <<8 x i16>> [#uses=1]
	%tmp10 = insertelement <8 x i16> %tmp9, i16 %tmp0, i32 2		; <<8 x i16>> [#uses=1]
	%tmp11 = insertelement <8 x i16> %tmp10, i16 %tmp3, i32 3		; <<8 x i16>> [#uses=1]
	%tmp12 = insertelement <8 x i16> %tmp11, i16 %tmp6, i32 4		; <<8 x i16>> [#uses=1]
	%tmp13 = insertelement <8 x i16> %tmp12, i16 %tmp5, i32 5		; <<8 x i16>> [#uses=1]
	%tmp14 = insertelement <8 x i16> %tmp13, i16 %tmp4, i32 6		; <<8 x i16>> [#uses=1]
	%tmp15 = insertelement <8 x i16> %tmp14, i16 %tmp7, i32 7		; <<8 x i16>> [#uses=1]
	%tmp15.upgrd.2 = bitcast <8 x i16> %tmp15 to <2 x i64>		; <<2 x i64>> [#uses=1]
	store <2 x i64> %tmp15.upgrd.2, <2 x i64>* %res
	ret void
}

define void @test2(<4 x float>* %r, <2 x i32>* %A) nounwind {
	%tmp = load <4 x float>* %r		; <<4 x float>> [#uses=2]
	%tmp.upgrd.3 = bitcast <2 x i32>* %A to double*		; <double*> [#uses=1]
	%tmp.upgrd.4 = load double* %tmp.upgrd.3		; <double> [#uses=1]
	%tmp.upgrd.5 = insertelement <2 x double> undef, double %tmp.upgrd.4, i32 0		; <<2 x double>> [#uses=1]
	%tmp5 = insertelement <2 x double> %tmp.upgrd.5, double undef, i32 1		; <<2 x double>> [#uses=1]
	%tmp6 = bitcast <2 x double> %tmp5 to <4 x float>		; <<4 x float>> [#uses=2]
	%tmp.upgrd.6 = extractelement <4 x float> %tmp, i32 0		; <float> [#uses=1]
	%tmp7 = extractelement <4 x float> %tmp, i32 1		; <float> [#uses=1]
	%tmp8 = extractelement <4 x float> %tmp6, i32 0		; <float> [#uses=1]
	%tmp9 = extractelement <4 x float> %tmp6, i32 1		; <float> [#uses=1]
	%tmp10 = insertelement <4 x float> undef, float %tmp.upgrd.6, i32 0		; <<4 x float>> [#uses=1]
	%tmp11 = insertelement <4 x float> %tmp10, float %tmp7, i32 1		; <<4 x float>> [#uses=1]
	%tmp12 = insertelement <4 x float> %tmp11, float %tmp8, i32 2		; <<4 x float>> [#uses=1]
	%tmp13 = insertelement <4 x float> %tmp12, float %tmp9, i32 3		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp13, <4 x float>* %r
	ret void
}
