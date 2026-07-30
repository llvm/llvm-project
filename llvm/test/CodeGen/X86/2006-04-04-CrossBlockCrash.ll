; RUN: llc < %s -mcpu=yonah
; END.

target datalayout = "e-p:32:32"
target triple = "i686-apple-darwin8.6.1"
	%struct.GLTColor4 = type { float, float, float, float }
	%struct.GLTCoord3 = type { float, float, float }
	%struct.__GLIContextRec = type { { %struct.anon, { [24 x [16 x float]], [24 x [16 x float]] }, %struct.GLTColor4, { float, float, float, float, %struct.GLTCoord3, float } }, { float, float, float, float, float, float, float, float, [4 x i32], [4 x i32], [4 x i32] } }
	%struct.__GLvertex = type { %struct.GLTColor4, %struct.GLTColor4, %struct.GLTColor4, %struct.GLTColor4, %struct.GLTColor4, %struct.GLTCoord3, float, %struct.GLTColor4, float, float, float, i8, i8, i8, i8, [4 x float], [2 x ptr], i32, i32, [16 x %struct.GLTColor4] }
	%struct.anon = type { float, float, float, float, float, float, float, float }

declare <4 x float> @llvm.x86.sse.cmp.ps(<4 x float>, <4 x float>, i8)

declare <8 x i16> @llvm.x86.sse2.packssdw.128(<4 x i32>, <4 x i32>)

declare i32 @llvm.x86.sse2.pmovmskb.128(<16 x i8>)

define void @gleLLVMVecInterpolateClip() {
entry:
	br i1 false, label %cond_false, label %cond_false183
cond_false:		; preds = %entry
	br i1 false, label %cond_false183, label %cond_true69
cond_true69:		; preds = %cond_false
	ret void
cond_false183:		; preds = %cond_false, %entry
	%vuizmsk.0.1 = phi <4 x i32> [ < i32 -1, i32 -1, i32 -1, i32 0 >, %entry ], [ < i32 -1, i32 0, i32 0, i32 0 >, %cond_false ]		; <<4 x i32>> [#uses=2]
	%tmp192 = extractelement <4 x i32> %vuizmsk.0.1, i32 2		; <i32> [#uses=1]
	%tmp193 = extractelement <4 x i32> %vuizmsk.0.1, i32 3		; <i32> [#uses=2]
	%tmp195 = insertelement <4 x i32> zeroinitializer, i32 %tmp192, i32 1		; <<4 x i32>> [#uses=1]
	%tmp196 = insertelement <4 x i32> %tmp195, i32 %tmp193, i32 2		; <<4 x i32>> [#uses=1]
	%tmp197 = insertelement <4 x i32> %tmp196, i32 %tmp193, i32 3		; <<4 x i32>> [#uses=1]
	%tmp336 = and <4 x i32> zeroinitializer, %tmp197		; <<4 x i32>> [#uses=1]
	%tmp337 = bitcast <4 x i32> %tmp336 to <4 x float>		; <<4 x float>> [#uses=1]
	%tmp378 = tail call <4 x float> @llvm.x86.sse.cmp.ps( <4 x float> %tmp337, <4 x float> zeroinitializer, i8 1 )		; <<4 x float>> [#uses=1]
	%tmp379 = bitcast <4 x float> %tmp378 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp388 = tail call <8 x i16> @llvm.x86.sse2.packssdw.128( <4 x i32> zeroinitializer, <4 x i32> %tmp379 )		; <<4 x i32>> [#uses=1]
	%tmp392 = bitcast <8 x i16> %tmp388 to <8 x i16>		; <<8 x i16>> [#uses=1]
	%tmp399 = extractelement <8 x i16> %tmp392, i32 7		; <i16> [#uses=1]
	%tmp423 = insertelement <8 x i16> zeroinitializer, i16 %tmp399, i32 7		; <<8 x i16>> [#uses=1]
	%tmp427 = bitcast <8 x i16> %tmp423 to <16 x i8>		; <<16 x i8>> [#uses=1]
	%tmp428 = tail call i32 @llvm.x86.sse2.pmovmskb.128( <16 x i8> %tmp427 )		; <i32> [#uses=1]
	%tmp432 = trunc i32 %tmp428 to i8		; <i8> [#uses=1]
	%tmp = and i8 %tmp432, 42		; <i8> [#uses=1]
	%tmp436 = bitcast i8 %tmp to i8		; <i8> [#uses=1]
	%tmp446 = zext i8 %tmp436 to i32		; <i32> [#uses=1]
	%tmp447 = shl i32 %tmp446, 24		; <i32> [#uses=1]
	%tmp449 = or i32 0, %tmp447		; <i32> [#uses=1]
	store i32 %tmp449, ptr null
	ret void
}
