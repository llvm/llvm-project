; RUN: llc < %s -march=x86 -mcpu=penryn -mattr=+sse2 | grep pshufd
; RUN: llc < %s -march=x86 -mcpu=penryn -mattr=+sse3 | grep movddup

define void @test_v4sf(<4 x float>* %P, <4 x float>* %Q, float %X) nounwind {
	%tmp = insertelement <4 x float> zeroinitializer, float %X, i32 0		; <<4 x float>> [#uses=1]
	%tmp2 = insertelement <4 x float> %tmp, float %X, i32 1		; <<4 x float>> [#uses=1]
	%tmp4 = insertelement <4 x float> %tmp2, float %X, i32 2		; <<4 x float>> [#uses=1]
	%tmp6 = insertelement <4 x float> %tmp4, float %X, i32 3		; <<4 x float>> [#uses=1]
	%tmp8 = load <4 x float>* %Q		; <<4 x float>> [#uses=1]
	%tmp10 = fmul <4 x float> %tmp8, %tmp6		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp10, <4 x float>* %P
	ret void
}

define void @test_v2sd(<2 x double>* %P, <2 x double>* %Q, double %X) nounwind {
	%tmp = insertelement <2 x double> zeroinitializer, double %X, i32 0		; <<2 x double>> [#uses=1]
	%tmp2 = insertelement <2 x double> %tmp, double %X, i32 1		; <<2 x double>> [#uses=1]
	%tmp4 = load <2 x double>* %Q		; <<2 x double>> [#uses=1]
	%tmp6 = fmul <2 x double> %tmp4, %tmp2		; <<2 x double>> [#uses=1]
	store <2 x double> %tmp6, <2 x double>* %P
	ret void
}
