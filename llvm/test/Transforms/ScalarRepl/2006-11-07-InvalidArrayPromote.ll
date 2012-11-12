; RUN: opt < %s -scalarrepl -S | not grep alloca
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"

define i32 @func(<4 x float> %v0, <4 x float> %v1) nounwind {
	%vsiidx = alloca [2 x <4 x i32>], align 16		; <[2 x <4 x i32>]*> [#uses=3]
	%tmp = call <4 x i32> @llvm.x86.sse2.cvttps2dq( <4 x float> %v0 )		; <<4 x i32>> [#uses=2]
	%tmp.upgrd.1 = bitcast <4 x i32> %tmp to <2 x i64>		; <<2 x i64>> [#uses=0]
	%tmp.upgrd.2 = getelementptr [2 x <4 x i32>]* %vsiidx, i32 0, i32 0		; <<4 x i32>*> [#uses=1]
	store <4 x i32> %tmp, <4 x i32>* %tmp.upgrd.2
	%tmp10 = call <4 x i32> @llvm.x86.sse2.cvttps2dq( <4 x float> %v1 )		; <<4 x i32>> [#uses=2]
	%tmp10.upgrd.3 = bitcast <4 x i32> %tmp10 to <2 x i64>		; <<2 x i64>> [#uses=0]
	%tmp14 = getelementptr [2 x <4 x i32>]* %vsiidx, i32 0, i32 1		; <<4 x i32>*> [#uses=1]
	store <4 x i32> %tmp10, <4 x i32>* %tmp14
	%tmp15 = getelementptr [2 x <4 x i32>]* %vsiidx, i32 0, i32 0, i32 4		; <i32*> [#uses=1]
	%tmp.upgrd.4 = load i32* %tmp15		; <i32> [#uses=1]
	ret i32 %tmp.upgrd.4
}

declare <4 x i32> @llvm.x86.sse2.cvttps2dq(<4 x float>)

