; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2

define <4 x float> @opRSQ(<4 x float> %a) nounwind {
entry:
	%tmp2 = extractelement <4 x float> %a, i32 3		; <float> [#uses=2]
	%abscond = fcmp oge float %tmp2, -0.000000e+00		; <i1> [#uses=1]
	%abs = select i1 %abscond, float %tmp2, float 0.000000e+00		; <float> [#uses=1]
	%tmp3 = tail call float @llvm.sqrt.f32( float %abs )		; <float> [#uses=1]
	%tmp4 = fdiv float 1.000000e+00, %tmp3		; <float> [#uses=1]
	%tmp11 = insertelement <4 x float> zeroinitializer, float %tmp4, i32 3		; <<4 x float>> [#uses=1]
	ret <4 x float> %tmp11
}

declare float @llvm.sqrt.f32(float)

