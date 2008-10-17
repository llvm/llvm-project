; RUN: llvm-as < %s | llc -relocation-model=pic -disable-fp-elim -mtriple=i386-apple-darwin | grep and | grep 7 | grep ebp

	%struct.XXDActiveTextureTargets = type { i64, i64, i64, i64, i64, i64 }
	%struct.XXDAlphaTest = type { float, i16, i8, i8 }
	%struct.XXDArrayRange = type { i8, i8, i8, i8 }
	%struct.XXDBlendMode = type { i16, i16, i16, i16, %struct.XXTColor4, i16, i16, i8, i8, i8, i8 }
	%struct.XXDClearColor = type { double, %struct.XXTColor4, %struct.XXTColor4, float, i32 }
	%struct.XXDClipPlane = type { i32, [6 x %struct.XXTColor4] }
	%struct.XXDColorBuffer = type { i16, i8, i8, [8 x i16], i8, i8, i8, i8 }
	%struct.XXDColorMatrix = type { [16 x float]*, %struct.XXDImagingCC }
	%struct.XXDConvolution = type { %struct.XXTColor4, %struct.XXDImagingCC, i16, i16, [0 x i32], float*, i32, i32 }
	%struct.XXDDepthTest = type { i16, i16, i8, i8, i8, i8, double, double }
	%struct.XXDFixedFunction = type { %struct.YYToken* }
	%struct.XXDFogMode = type { %struct.XXTColor4, float, float, float, float, float, i16, i16, i16, i8, i8 }
	%struct.XXDHintMode = type { i16, i16, i16, i16, i16, i16, i16, i16, i16, i16 }
	%struct.XXDHistogram = type { %struct.XXTFixedColor4*, i32, i16, i8, i8 }
	%struct.XXDImagingCC = type { { float, float }, { float, float }, { float, float }, { float, float } }
	%struct.XXDImagingSubset = type { %struct.XXDConvolution, %struct.XXDConvolution, %struct.XXDConvolution, %struct.XXDColorMatrix, %struct.XXDMinmax, %struct.XXDHistogram, %struct.XXDImagingCC, %struct.XXDImagingCC, %struct.XXDImagingCC, %struct.XXDImagingCC, i32, [0 x i32] }
	%struct.XXDLight = type { %struct.XXTColor4, %struct.XXTColor4, %struct.XXTColor4, %struct.XXTColor4, %struct.XXTCoord3, float, float, float, float, float, %struct.XXTCoord3, float, %struct.XXTCoord3, float, %struct.XXTCoord3, float, float, float, float, float }
	%struct.XXDLightModel = type { %struct.XXTColor4, [8 x %struct.XXDLight], [2 x %struct.XXDMaterial], i32, i16, i16, i16, i8, i8, i8, i8, i8, i8 }
	%struct.XXDLightProduct = type { %struct.XXTColor4, %struct.XXTColor4, %struct.XXTColor4 }
	%struct.XXDLineMode = type { float, i32, i16, i16, i8, i8, i8, i8 }
	%struct.XXDLogicOp = type { i16, i8, i8 }
	%struct.XXDMaskMode = type { i32, [3 x i32], i8, i8, i8, i8, i8, i8, i8, i8 }
	%struct.XXDMaterial = type { %struct.XXTColor4, %struct.XXTColor4, %struct.XXTColor4, %struct.XXTColor4, float, float, float, float, [8 x %struct.XXDLightProduct], %struct.XXTColor4, [8 x i32] }
	%struct.XXDMinmax = type { %struct.XXDMinmaxTable*, i16, i8, i8, [0 x i32] }
	%struct.XXDMinmaxTable = type { %struct.XXTColor4, %struct.XXTColor4 }
	%struct.XXDMultisample = type { float, i8, i8, i8, i8, i8, i8, i8, i8 }
	%struct.XXDPipelineProgramState = type { i8, i8, i8, i8, [0 x i32], %struct.XXTColor4* }
	%struct.XXDPixelMap = type { i32*, float*, float*, float*, float*, float*, float*, float*, float*, i32*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.XXDPixelMode = type { float, float, %struct.XXDPixelStore, %struct.XXDPixelTransfer, %struct.XXDPixelMap, %struct.XXDImagingSubset, i32, i32 }
	%struct.XXDPixelPack = type { i32, i32, i32, i32, i32, i32, i32, i32, i8, i8, i8, i8 }
	%struct.XXDPixelStore = type { %struct.XXDPixelPack, %struct.XXDPixelPack }
	%struct.XXDPixelTransfer = type { float, float, float, float, float, float, float, float, float, float, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float }
	%struct.XXDPointMode = type { float, float, float, float, %struct.XXTCoord3, float, i8, i8, i8, i8, i16, i16, i32, i16, i16 }
	%struct.XXDPolygonMode = type { [128 x i8], float, float, i16, i16, i16, i16, i8, i8, i8, i8, i8, i8, i8, i8 }
	%struct.XXDRegisterCombiners = type { i8, i8, i8, i8, i32, [2 x %struct.XXTColor4], [8 x %struct.XXDRegisterCombinersPerStageState], %struct.XXDRegisterCombinersFinalStageState }
	%struct.XXDRegisterCombinersFinalStageState = type { i8, i8, i8, i8, [7 x %struct.XXDRegisterCombinersPerVariableState] }
	%struct.XXDRegisterCombinersPerPortionState = type { [4 x %struct.XXDRegisterCombinersPerVariableState], i8, i8, i8, i8, i16, i16, i16, i16, i16, i16 }
	%struct.XXDRegisterCombinersPerStageState = type { [2 x %struct.XXDRegisterCombinersPerPortionState], [2 x %struct.XXTColor4] }
	%struct.XXDRegisterCombinersPerVariableState = type { i16, i16, i16, i16 }
	%struct.XXDScissorTest = type { %struct.XXTFixedColor4, i8, i8, i8, i8 }
	%struct.XXDState = type <{ i16, i16, i16, i16, i32, i32, [256 x %struct.XXTColor4], [128 x %struct.XXTColor4], %struct.XXDViewport, %struct.XXDTransform, %struct.XXDLightModel, %struct.XXDActiveTextureTargets, %struct.XXDAlphaTest, %struct.XXDBlendMode, %struct.XXDClearColor, %struct.XXDColorBuffer, %struct.XXDDepthTest, %struct.XXDArrayRange, %struct.XXDFogMode, %struct.XXDHintMode, %struct.XXDLineMode, %struct.XXDLogicOp, %struct.XXDMaskMode, %struct.XXDPixelMode, %struct.XXDPointMode, %struct.XXDPolygonMode, %struct.XXDScissorTest, i32, %struct.XXDStencilTest, [8 x %struct.XXDTextureMode], [16 x %struct.XXDTextureImageMode], %struct.XXDArrayRange, [8 x %struct.XXDTextureCoordGen], %struct.XXDClipPlane, %struct.XXDMultisample, %struct.XXDRegisterCombiners, %struct.XXDArrayRange, %struct.XXDArrayRange, [3 x %struct.XXDPipelineProgramState], %struct.XXDArrayRange, %struct.XXDTransformFeedback, i32*, %struct.XXDFixedFunction, [3 x i32], [2 x i32] }>
	%struct.XXDStencilTest = type { [3 x { i32, i32, i16, i16, i16, i16 }], i32, [4 x i8] }
	%struct.XXDTextureCoordGen = type { { i16, i16, %struct.XXTColor4, %struct.XXTColor4 }, { i16, i16, %struct.XXTColor4, %struct.XXTColor4 }, { i16, i16, %struct.XXTColor4, %struct.XXTColor4 }, { i16, i16, %struct.XXTColor4, %struct.XXTColor4 }, i8, i8, i8, i8 }
	%struct.XXDTextureImageMode = type { float }
	%struct.XXDTextureMode = type { %struct.XXTColor4, i32, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, float, float, i16, i16, i16, i16, i16, i16, [4 x i16], i8, i8, i8, i8, [3 x float], [4 x float], float, float }
	%struct.XXDTextureRec = type opaque
	%struct.XXDTransform = type <{ [24 x [16 x float]], [24 x [16 x float]], [16 x float], float, float, float, float, float, i8, i8, i8, i8, i32, i32, i32, i16, i16, i8, i8, i8, i8, i32 }>
	%struct.XXDTransformFeedback = type { i8, i8, i8, i8, [0 x i32], [16 x i32], [16 x i32] }
	%struct.XXDViewport = type { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, double, double, i32, i32, i32, i32, float, float, float, float }
	%struct.XXTColor4 = type { float, float, float, float }
	%struct.XXTCoord3 = type { float, float, float }
	%struct.XXTFixedColor4 = type { i32, i32, i32, i32 }
	%struct.XXVMTextures = type { [16 x %struct.XXDTextureRec*] }
	%struct.XXVMVPContext = type { i32 }
	%struct.XXVMVPStack = type { i32, i32 }
	%struct.YYToken = type { { i16, i16, i32 } }
	%struct._XXVMConstants = type { <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, float, float, float, float, float, float, float, float, float, float, float, float, [256 x float], [4096 x i8], [8 x float], [48 x float], [128 x float], [528 x i8], { void (i8*, i8*, i32, i8*)*, float (float)*, float (float)*, float (float)*, i32 (float)* } }
@llvm.used = appending global [1 x i8*] [ i8* bitcast (void (%struct.XXDState*, <4 x float>*, <4 x float>**, %struct._XXVMConstants*, %struct.YYToken*, %struct.XXVMVPContext*, %struct.XXVMTextures*, %struct.XXVMVPStack*, <4 x float>*, <4 x float>*, <4 x float>*, <4 x float>*, <4 x float>*, <4 x float>*, <4 x float>*, <4 x float>*, [4 x <4 x float>]*, i32*, <4 x i32>*, i64)* @t to i8*) ], section "llvm.metadata"		; <[1 x i8*]*> [#uses=0]

define void @t(%struct.XXDState* %gldst, <4 x float>* %prgrm, <4 x float>** %buffs, %struct._XXVMConstants* %cnstn, %struct.YYToken* %pstrm, %struct.XXVMVPContext* %vmctx, %struct.XXVMTextures* %txtrs, %struct.XXVMVPStack* %vpstk, <4 x float>* %atr0, <4 x float>* %atr1, <4 x float>* %atr2, <4 x float>* %atr3, <4 x float>* %vtx0, <4 x float>* %vtx1, <4 x float>* %vtx2, <4 x float>* %vtx3, [4 x <4 x float>]* %tmpGbl, i32* %oldMsk, <4 x i32>* %adrGbl, i64 %key_token) nounwind {
entry:
	%0 = trunc i64 %key_token to i32		; <i32> [#uses=1]
	%1 = getelementptr %struct.YYToken* %pstrm, i32 %0		; <%struct.YYToken*> [#uses=5]
	br label %bb1132

bb51:		; preds = %bb1132
	%2 = getelementptr %struct.YYToken* %1, i32 %operation.0.rec, i32 0, i32 0		; <i16*> [#uses=1]
	%3 = load i16* %2, align 1		; <i16> [#uses=3]
	%4 = lshr i16 %3, 6		; <i16> [#uses=1]
	%5 = trunc i16 %4 to i8		; <i8> [#uses=1]
	%6 = zext i8 %5 to i32		; <i32> [#uses=1]
	%7 = trunc i16 %3 to i8		; <i8> [#uses=1]
	%8 = and i8 %7, 7		; <i8> [#uses=1]
	%mask5556 = zext i8 %8 to i32		; <i32> [#uses=3]
	%.sum1324 = add i32 %mask5556, 2		; <i32> [#uses=1]
	%.rec = add i32 %operation.0.rec, %.sum1324		; <i32> [#uses=1]
	%9 = bitcast %struct.YYToken* %operation.0 to i32*		; <i32*> [#uses=1]
	%10 = load i32* %9, align 1		; <i32> [#uses=1]
	%11 = lshr i32 %10, 16		; <i32> [#uses=2]
	%12 = trunc i32 %11 to i8		; <i8> [#uses=1]
	%13 = and i8 %12, 1		; <i8> [#uses=1]
	%14 = lshr i16 %3, 15		; <i16> [#uses=1]
	%15 = trunc i16 %14 to i8		; <i8> [#uses=1]
	%16 = or i8 %13, %15		; <i8> [#uses=1]
	%17 = icmp eq i8 %16, 0		; <i1> [#uses=1]
	br i1 %17, label %bb94, label %bb75

bb75:		; preds = %bb51
	%18 = getelementptr %struct.YYToken* %1, i32 0, i32 0, i32 0		; <i16*> [#uses=1]
	%19 = load i16* %18, align 4		; <i16> [#uses=1]
	%20 = load i16* null, align 2		; <i16> [#uses=1]
	%21 = zext i16 %19 to i64		; <i64> [#uses=1]
	%22 = zext i16 %20 to i64		; <i64> [#uses=1]
	%23 = shl i64 %22, 16		; <i64> [#uses=1]
	%.ins1177 = or i64 %23, %21		; <i64> [#uses=1]
	%.ins1175 = or i64 %.ins1177, 0		; <i64> [#uses=1]
	%24 = and i32 %11, 1		; <i32> [#uses=1]
	%.neg1333 = sub i32 %mask5556, %24		; <i32> [#uses=1]
	%.neg1335 = sub i32 %.neg1333, 0		; <i32> [#uses=1]
	%25 = sub i32 %.neg1335, 0		; <i32> [#uses=1]
	br label %bb94

bb94:		; preds = %bb75, %bb51
	%extraToken.0 = phi i64 [ %.ins1175, %bb75 ], [ %extraToken.1, %bb51 ]		; <i64> [#uses=1]
	%argCount.0 = phi i32 [ %25, %bb75 ], [ %mask5556, %bb51 ]		; <i32> [#uses=1]
	%operation.0.sum1392 = add i32 %operation.0.rec, 1		; <i32> [#uses=2]
	%26 = getelementptr %struct.YYToken* %1, i32 %operation.0.sum1392, i32 0, i32 0		; <i16*> [#uses=1]
	%27 = load i16* %26, align 4		; <i16> [#uses=1]
	%28 = getelementptr %struct.YYToken* %1, i32 %operation.0.sum1392, i32 0, i32 1		; <i16*> [#uses=1]
	%29 = load i16* %28, align 2		; <i16> [#uses=1]
	store i16 %27, i16* null, align 8
	store i16 %29, i16* null, align 2
	br i1 false, label %bb1132, label %bb110

bb110:		; preds = %bb94
	switch i32 %6, label %bb1078 [
		i32 30, label %bb960
		i32 32, label %bb801
		i32 38, label %bb809
		i32 78, label %bb1066
	]

bb801:		; preds = %bb110
	unreachable

bb809:		; preds = %bb110
	unreachable

bb960:		; preds = %bb110
	%30 = icmp eq i32 %argCount.0, 1		; <i1> [#uses=1]
	br i1 %30, label %bb962, label %bb965

bb962:		; preds = %bb960
	unreachable

bb965:		; preds = %bb960
	unreachable

bb1066:		; preds = %bb110
	unreachable

bb1078:		; preds = %bb110
	unreachable

bb1132:		; preds = %bb94, %entry
	%extraToken.1 = phi i64 [ undef, %entry ], [ %extraToken.0, %bb94 ]		; <i64> [#uses=1]
	%operation.0.rec = phi i32 [ 0, %entry ], [ %.rec, %bb94 ]		; <i32> [#uses=4]
	%operation.0 = getelementptr %struct.YYToken* %1, i32 %operation.0.rec		; <%struct.YYToken*> [#uses=1]
	br i1 false, label %bb1134, label %bb51

bb1134:		; preds = %bb1132
	ret void
}
