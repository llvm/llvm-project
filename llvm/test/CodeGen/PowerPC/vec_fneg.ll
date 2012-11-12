; RUN: llc < %s -march=ppc32 -mcpu=g5 | grep vsubfp

define void @t(<4 x float>* %A) {
	%tmp2 = load <4 x float>* %A
	%tmp3 = fsub <4 x float> < float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00 >, %tmp2
	store <4 x float> %tmp3, <4 x float>* %A
	ret void
}
