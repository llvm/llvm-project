; RUN: llc < %s -mtriple=i686-linux -mcpu=core2 | FileCheck %s

; CHECK: test_v4sf
; CHECK: movq 8(%esp)
; CHECK: pshufd $80
define void @test_v4sf(<4 x float>* %P, float %X, float %Y) nounwind {
	%tmp = insertelement <4 x float> zeroinitializer, float %X, i32 0		; <<4 x float>> [#uses=1]
	%tmp2 = insertelement <4 x float> %tmp, float %X, i32 1		; <<4 x float>> [#uses=1]
	%tmp4 = insertelement <4 x float> %tmp2, float %Y, i32 2		; <<4 x float>> [#uses=1]
	%tmp6 = insertelement <4 x float> %tmp4, float %Y, i32 3		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp6, <4 x float>* %P
	ret void
}

; CHECK: test_v2sd
; CHECK: movups	8(%esp)
; CHECK: movaps
define void @test_v2sd(<2 x double>* %P, double %X, double %Y) nounwind {
	%tmp = insertelement <2 x double> zeroinitializer, double %X, i32 0		; <<2 x double>> [#uses=1]
	%tmp2 = insertelement <2 x double> %tmp, double %Y, i32 1		; <<2 x double>> [#uses=1]
	store <2 x double> %tmp2, <2 x double>* %P
	ret void
}

; CHECK: test_v8i16
; CHECK: pshufhw $-58
; CHECK: movdqa
define void @test_v8i16(<2 x i64>* %res, <2 x i64>* %A) nounwind {
	%tmp = load <2 x i64>* %A		; <<2 x i64>> [#uses=1]
	%tmp.upgrd.1 = bitcast <2 x i64> %tmp to <8 x i16>		; <<8 x i16>> [#uses=8]
	%tmp.upgrd.2 = extractelement <8 x i16> %tmp.upgrd.1, i32 0		; <i16> [#uses=1]
	%tmp1 = extractelement <8 x i16> %tmp.upgrd.1, i32 1		; <i16> [#uses=1]
	%tmp2 = extractelement <8 x i16> %tmp.upgrd.1, i32 2		; <i16> [#uses=1]
	%tmp3 = extractelement <8 x i16> %tmp.upgrd.1, i32 3		; <i16> [#uses=1]
	%tmp4 = extractelement <8 x i16> %tmp.upgrd.1, i32 6		; <i16> [#uses=1]
	%tmp5 = extractelement <8 x i16> %tmp.upgrd.1, i32 5		; <i16> [#uses=1]
	%tmp6 = extractelement <8 x i16> %tmp.upgrd.1, i32 4		; <i16> [#uses=1]
	%tmp7 = extractelement <8 x i16> %tmp.upgrd.1, i32 7		; <i16> [#uses=1]
	%tmp8 = insertelement <8 x i16> undef, i16 %tmp.upgrd.2, i32 0		; <<8 x i16>> [#uses=1]
	%tmp9 = insertelement <8 x i16> %tmp8, i16 %tmp1, i32 1		; <<8 x i16>> [#uses=1]
	%tmp10 = insertelement <8 x i16> %tmp9, i16 %tmp2, i32 2		; <<8 x i16>> [#uses=1]
	%tmp11 = insertelement <8 x i16> %tmp10, i16 %tmp3, i32 3		; <<8 x i16>> [#uses=1]
	%tmp12 = insertelement <8 x i16> %tmp11, i16 %tmp4, i32 4		; <<8 x i16>> [#uses=1]
	%tmp13 = insertelement <8 x i16> %tmp12, i16 %tmp5, i32 5		; <<8 x i16>> [#uses=1]
	%tmp14 = insertelement <8 x i16> %tmp13, i16 %tmp6, i32 6		; <<8 x i16>> [#uses=1]
	%tmp15 = insertelement <8 x i16> %tmp14, i16 %tmp7, i32 7		; <<8 x i16>> [#uses=1]
	%tmp15.upgrd.3 = bitcast <8 x i16> %tmp15 to <2 x i64>		; <<2 x i64>> [#uses=1]
	store <2 x i64> %tmp15.upgrd.3, <2 x i64>* %res
	ret void
}
