; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2

define <4 x float> @t1(<4 x float> %Q) {
        %tmp15 = sub <4 x float> < float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00 >, %Q
	ret <4 x float> %tmp15
}

define <4 x float> @t2(<4 x float> %Q) {
        %tmp15 = sub <4 x float> zeroinitializer, %Q
	ret <4 x float> %tmp15
}
