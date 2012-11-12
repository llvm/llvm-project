; RUN: llc < %s -march=x86 -mcpu=yonah | grep 1084227584 | count 1

; ModuleID = '<stdin>'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin9.2.2"

define <8 x float> @f(<8 x float> %a, i32 %b) nounwind  {
entry:
	%vecins = insertelement <8 x float> %a, float 5.000000e+00, i32 %b		; <<4 x float>> [#uses=1]
	ret <8 x float> %vecins
}
