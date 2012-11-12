; RUN: opt < %s -instcombine -S | grep "ret <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0x7FF0000000000000, float 0x7FF0000000000000>"

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9"

define <4 x float> @__inff4() nounwind readnone {
entry:
	%tmp14 = extractelement <1 x double> bitcast (<2 x float> <float 0x7FF0000000000000, float 0x7FF0000000000000> to <1 x double>), i32 0		; <double> [#uses=1]
	%tmp4 = bitcast double %tmp14 to i64		; <i64> [#uses=1]
	%tmp3 = bitcast i64 %tmp4 to <2 x float>		; <<2 x float>> [#uses=1]
	%tmp8 = shufflevector <2 x float> %tmp3, <2 x float> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>		; <<4 x float>> [#uses=1]
	%tmp9 = shufflevector <4 x float> zeroinitializer, <4 x float> %tmp8, <4 x i32> <i32 0, i32 1, i32 4, i32 5>		; <<4 x float>> [#uses=0]
	ret <4 x float> %tmp9
}
