; RUN: opt < %s -scalarrepl -S | \
; RUN:   not grep alloca
; RUN: opt < %s -scalarrepl -S | \
; RUN:   grep bitcast
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"

define <4 x i32> @test(<4 x float> %X) {
	%X_addr = alloca <4 x float>		; <<4 x float>*> [#uses=2]
	store <4 x float> %X, <4 x float>* %X_addr
	%X_addr.upgrd.1 = bitcast <4 x float>* %X_addr to <4 x i32>*		; <<4 x i32>*> [#uses=1]
	%tmp = load <4 x i32>* %X_addr.upgrd.1		; <<4 x i32>> [#uses=1]
	ret <4 x i32> %tmp
}

