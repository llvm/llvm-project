; RUN: llc < %s -mattr=+neon
; PR4657

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:64"
target triple = "armv7-apple-darwin9"

define <4 x i32> @scale(<4 x i32> %v, i32 %f) nounwind {
entry:
	%v_addr = alloca <4 x i32>		; <<4 x i32>*> [#uses=2]
	%f_addr = alloca i32		; <i32*> [#uses=2]
	%retval = alloca <4 x i32>		; <<4 x i32>*> [#uses=2]
	%0 = alloca <4 x i32>		; <<4 x i32>*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store <4 x i32> %v, <4 x i32>* %v_addr
	store i32 %f, i32* %f_addr
	%1 = load <4 x i32>, <4 x i32>* %v_addr, align 16		; <<4 x i32>> [#uses=1]
	%2 = load i32, i32* %f_addr, align 4		; <i32> [#uses=1]
	%3 = insertelement <4 x i32> undef, i32 %2, i32 0		; <<4 x i32>> [#uses=1]
	%4 = shufflevector <4 x i32> %3, <4 x i32> undef, <4 x i32> zeroinitializer		; <<4 x i32>> [#uses=1]
	%5 = mul <4 x i32> %1, %4		; <<4 x i32>> [#uses=1]
	store <4 x i32> %5, <4 x i32>* %0, align 16
	%6 = load <4 x i32>, <4 x i32>* %0, align 16		; <<4 x i32>> [#uses=1]
	store <4 x i32> %6, <4 x i32>* %retval, align 16
	br label %return

return:		; preds = %entry
	%retval1 = load <4 x i32>, <4 x i32>* %retval		; <<4 x i32>> [#uses=1]
	ret <4 x i32> %retval1
}
