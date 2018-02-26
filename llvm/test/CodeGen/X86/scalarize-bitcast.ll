; RUN: llc < %s
; PR3886

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "x86_64-pc-linux-gnu"

define void @mmxCombineMaskU(i32* nocapture %src, i32* nocapture %mask) nounwind {
entry:
	%tmp1 = load i32, i32* %src		; <i32> [#uses=1]
	%0 = insertelement <2 x i32> undef, i32 %tmp1, i32 0		; <<2 x i32>> [#uses=1]
	%1 = insertelement <2 x i32> %0, i32 0, i32 1		; <<2 x i32>> [#uses=1]
	%conv.i.i = bitcast <2 x i32> %1 to <1 x i64>		; <<1 x i64>> [#uses=1]
	%tmp2.i.i = extractelement <1 x i64> %conv.i.i, i32 0		; <i64> [#uses=1]
	%tmp22.i = bitcast i64 %tmp2.i.i to <1 x i64>		; <<1 x i64>> [#uses=1]
	%tmp15.i = extractelement <1 x i64> %tmp22.i, i32 0		; <i64> [#uses=1]
	%conv.i26.i = bitcast i64 %tmp15.i to <8 x i8>		; <<8 x i8>> [#uses=1]
	%shuffle.i.i = shufflevector <8 x i8> %conv.i26.i, <8 x i8> <i8 0, i8 0, i8 0, i8 0, i8 undef, i8 undef, i8 undef, i8 undef>, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>		; <<8 x i8>> [#uses=1]
	%conv6.i.i = bitcast <8 x i8> %shuffle.i.i to <1 x i64>		; <<1 x i64>> [#uses=1]
	%tmp12.i.i = extractelement <1 x i64> %conv6.i.i, i32 0		; <i64> [#uses=1]
	%tmp10.i = bitcast i64 %tmp12.i.i to <1 x i64>		; <<1 x i64>> [#uses=1]
	%tmp24.i = extractelement <1 x i64> %tmp10.i, i32 0		; <i64> [#uses=1]
	%tmp10 = bitcast i64 %tmp24.i to <1 x i64>		; <<1 x i64>> [#uses=1]
	%tmp7 = extractelement <1 x i64> %tmp10, i32 0		; <i64> [#uses=1]
	%call6 = tail call i32 (...) @store8888(i64 %tmp7)		; <i32> [#uses=1]
	store i32 %call6, i32* %src
	ret void
}

declare i32 @store8888(...)
