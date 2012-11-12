; RUN: llc < %s -march=x86 -mattr=+sse2 | grep movss

define fastcc void @t(<4 x float> %A) nounwind  {
	%tmp41896 = extractelement <4 x float> %A, i32 0		; <float> [#uses=1]
	%tmp14082 = insertelement <4 x float> < float 0.000000e+00, float undef, float undef, float undef >, float %tmp41896, i32 1		; <<4 x float>> [#uses=1]
	%tmp14083 = insertelement <4 x float> %tmp14082, float 0.000000e+00, i32 2		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp14083, <4 x float>* null, align 16
        ret void
}
