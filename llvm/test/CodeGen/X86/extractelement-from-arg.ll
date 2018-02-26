; RUN: llc < %s -mtriple=x86_64-- -mattr=+sse2

define void @test(float* %R, <4 x float> %X) nounwind {
	%tmp = extractelement <4 x float> %X, i32 3
	store float %tmp, float* %R
	ret void
}
