; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | not grep pslldq

define void @t() nounwind  {
entry:
	%tmp1 = shufflevector <4 x float> zeroinitializer, <4 x float> < float 0.000000e+00, float 1.000000e+00, float 0.000000e+00, float 1.000000e+00 >, <4 x i32> < i32 0, i32 1, i32 4, i32 5 >
	%tmp2 = insertelement <4 x float> %tmp1, float 1.000000e+00, i32 3
	store <4 x float> %tmp2, <4 x float>* null, align 16
	unreachable
}
