; RUN: llc < %s -mtriple=i686-- -mattr=+sse2

define fastcc void @glgVectorFloatConversion() nounwind  {
	%tmp12745 = load <4 x float>, ptr null, align 16		; <<4 x float>> [#uses=1]
	%tmp12773 = insertelement <4 x float> %tmp12745, float 1.000000e+00, i32 1		; <<4 x float>> [#uses=1]
	%tmp12774 = insertelement <4 x float> %tmp12773, float 0.000000e+00, i32 2		; <<4 x float>> [#uses=1]
	%tmp12775 = insertelement <4 x float> %tmp12774, float 1.000000e+00, i32 3		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp12775, ptr null, align 16
	unreachable
}
