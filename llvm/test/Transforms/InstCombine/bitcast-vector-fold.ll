; RUN: opt < %s -instcombine -S | not grep bitcast
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"

define <2 x i64> @test1() {
	%tmp3 = bitcast <4 x i32> < i32 0, i32 1, i32 2, i32 3 > to <2 x i64>
	ret <2 x i64> %tmp3
}

define <4 x i32> @test2() {
	%tmp3 = bitcast <2 x i64> < i64 0, i64 1 > to <4 x i32>
	ret <4 x i32> %tmp3
}

define <2 x double> @test3() {
	%tmp3 = bitcast <4 x i32> < i32 0, i32 1, i32 2, i32 3 > to <2 x double>
	ret <2 x double> %tmp3
}

define <4 x float> @test4() {
	%tmp3 = bitcast <2 x i64> < i64 0, i64 1 > to <4 x float>
	ret <4 x float> %tmp3
}

define <2 x i64> @test5() {
	%tmp3 = bitcast <4 x float> <float 0.0, float 1.0, float 2.0, float 3.0> to <2 x i64>
	ret <2 x i64> %tmp3
}

define <4 x i32> @test6() {
	%tmp3 = bitcast <2 x double> <double 0.5, double 1.0> to <4 x i32>
	ret <4 x i32> %tmp3
}
