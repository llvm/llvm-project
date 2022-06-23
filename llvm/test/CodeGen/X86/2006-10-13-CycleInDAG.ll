; RUN: llc < %s -mtriple=i686--
@str = external dso_local global [18 x i8]		; <ptr> [#uses=1]

define void @test() {
bb.i:
	%tmp.i660 = load <4 x float>, ptr null		; <<4 x float>> [#uses=1]
	call void (i32, ...) @printf( i32 0, ptr @str, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00 )
	%tmp152.i = load <4 x i32>, ptr null		; <<4 x i32>> [#uses=1]
	%tmp156.i = bitcast <4 x i32> %tmp152.i to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp175.i = bitcast <4 x float> %tmp.i660 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp176.i = xor <4 x i32> %tmp156.i, < i32 -1, i32 -1, i32 -1, i32 -1 >		; <<4 x i32>> [#uses=1]
	%tmp177.i = and <4 x i32> %tmp176.i, %tmp175.i		; <<4 x i32>> [#uses=1]
	%tmp190.i = or <4 x i32> %tmp177.i, zeroinitializer		; <<4 x i32>> [#uses=1]
	%tmp191.i = bitcast <4 x i32> %tmp190.i to <4 x float>		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp191.i, ptr null
	ret void
}

declare void @printf(i32, ...)
