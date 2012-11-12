; RUN: llc < %s -mtriple=x86_64-pc-mingw64 | grep movabsq

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define <4 x float> @RecursiveTestFunc1(i8*) {
EntryBlock:
	%1 = call <4 x float> inttoptr (i64 5367207198 to <4 x float> (i8*, float, float, float, float)*)(i8* %0, float 8.000000e+00, float 5.000000e+00, float 3.000000e+00, float 4.000000e+00)		; <<4 x float>> [#uses=1]
	ret <4 x float> %1
}
