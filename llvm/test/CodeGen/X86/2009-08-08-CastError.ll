; RUN: llc < %s -mtriple=x86_64-pc-mingw64 | FileCheck %s

; CHECK: movabsq

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define <4 x float> @RecursiveTestFunc1(ptr) {
EntryBlock:
	%1 = call <4 x float> inttoptr (i64 5367207198 to ptr)(ptr %0, float 8.000000e+00, float 5.000000e+00, float 3.000000e+00, float 4.000000e+00)		; <<4 x float>> [#uses=1]
	ret <4 x float> %1
}
