; RUN: llc < %s -mcpu=yonah
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i686-apple-darwin8"

define void @test(<4 x float> %tmp42i) {
	%tmp42 = call <4 x float> asm "movss $1, $0", "=x,m,~{dirflag},~{fpsr},~{flags}"( float* null )		; <<4 x float>> [#uses=1]
	%tmp49 = shufflevector <4 x float> %tmp42, <4 x float> undef, <4 x i32> zeroinitializer		; <<4 x float>> [#uses=1]
	br label %bb

bb:		; preds = %bb, %cond_true10
	%tmp52 = bitcast <4 x float> %tmp49 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp53 = call <4 x i32> @llvm.x86.sse2.psll.d( <4 x i32> %tmp52, <4 x i32> < i32 8, i32 undef, i32 undef, i32 undef > )		; <<4 x i32>> [#uses=1]
	%tmp105 = bitcast <4 x i32> %tmp53 to <4 x float>		; <<4 x float>> [#uses=1]
	%tmp108 = fsub <4 x float> zeroinitializer, %tmp105		; <<4 x float>> [#uses=0]
	br label %bb

return:		; preds = %entry
	ret void
}

declare <4 x i32> @llvm.x86.sse2.psll.d(<4 x i32>, <4 x i32>)
