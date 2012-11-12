; RUN: llc < %s -march=x86 -mattr=+sse,-sse2

define <4 x float> @f(float %w) nounwind {
entry:
	%retval = alloca <4 x float>		; <<4 x float>*> [#uses=2]
	%w.addr = alloca float		; <float*> [#uses=2]
	%.compoundliteral = alloca <4 x float>		; <<4 x float>*> [#uses=2]
	store float %w, float* %w.addr
	%tmp = load float* %w.addr		; <float> [#uses=1]
	%0 = insertelement <4 x float> undef, float %tmp, i32 0		; <<4 x float>> [#uses=1]
	%1 = insertelement <4 x float> %0, float 0.000000e+00, i32 1		; <<4 x float>> [#uses=1]
	%2 = insertelement <4 x float> %1, float 0.000000e+00, i32 2		; <<4 x float>> [#uses=1]
	%3 = insertelement <4 x float> %2, float 0.000000e+00, i32 3		; <<4 x float>> [#uses=1]
	store <4 x float> %3, <4 x float>* %.compoundliteral
	%tmp1 = load <4 x float>* %.compoundliteral		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp1, <4 x float>* %retval
	br label %return

return:		; preds = %entry
	%4 = load <4 x float>* %retval		; <<4 x float>> [#uses=1]
	ret <4 x float> %4
}
