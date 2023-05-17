; RUN: llc < %s -mtriple=i686-- -mattr=+sse2

define void @test(ptr %arg) {
	%tmp89 = getelementptr <4 x float>, ptr %arg, i64 3
	%tmp1144 = fsub <4 x float> < float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00 >, zeroinitializer
	store <4 x float> %tmp1144, ptr null
	%tmp1149 = load <4 x float>, ptr %tmp89
	%tmp1150 = fsub <4 x float> < float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00 >, %tmp1149
	store <4 x float> %tmp1150, ptr %tmp89
	ret void
}
