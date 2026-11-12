; Test VectorType handling by SCCP.
; SCCP ignores VectorTypes until PR 1034 is fixed
;
; RUN: opt < %s -passes=sccp
; END.

target datalayout = "E-p:32:32"
target triple = "powerpc-unknown-linux-gnu"
	%struct.GLDAlphaTest = type { float, i16, i8, i8 }
	%struct.GLDArrayRange = type { i8, i8, i8, i8 }
	%struct.GLDBlendMode = type { i16, i16, i16, i16, %struct.GLTColor4, i16, i16, i8, i8, i8, i8 }
	%struct.GLDBufferRec = type opaque
	%struct.GLDBufferstate = type { %struct.GLTDimensions, %struct.GLTDimensions, %struct.GLTFixedColor4, %struct.GLTFixedColor4, i8, i8, i8, i8, [2 x %struct.GLSBuffer], [4 x %struct.GLSBuffer], %struct.GLSBuffer, %struct.GLSBuffer, %struct.GLSBuffer, [4 x ptr], ptr, ptr, ptr, i8, i8 }
	%struct.GLDClearColor = type { double, %struct.GLTColor4, %struct.GLTColor4, float, i32 }
	%struct.GLDClipPlane = type { i32, [6 x %struct.GLTColor4] }
	%struct.GLDColorBuffer = type { i16, i16, [4 x i16] }
	%struct.GLDColorMatrix = type { ptr, %struct.GLDImagingColorScale }
	%struct.GLDContextRec = type { float, float, float, float, float, float, float, float, %struct.GLTColor4, %struct.GLTColor4, %struct.GLVMFPContext, %struct.GLDTextureMachine, %struct.GLGProcessor, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, float, float, float, i32, %struct.GLSDrawable, %struct.GLDFramebufferAttachment, %struct.GLDFormat, %struct.GLDBufferstate, ptr, ptr, ptr, %struct.GLTDimensions, ptr, ptr, ptr, ptr, ptr, ptr, %struct.GLDStateProgramRec, %struct.GLVMTextures, { [4 x ptr], ptr, ptr }, [64 x float], %struct.GLDStippleData, i16, i8, i8, i32, ptr, i8, ptr, ptr }
	%struct.GLDConvolution = type { %struct.GLTColor4, %struct.GLDImagingColorScale, i16, i16, ptr, i32, i32 }
	%struct.GLDDepthTest = type { i16, i16, i8, i8, i8, i8, double, double }
	%struct.GLDFogMode = type { %struct.GLTColor4, float, float, float, float, float, i16, i16, i16, i8, i8 }
	%struct.GLDFormat = type { i32, i32, i32, i32, i32, i32, i32, i32, i8, i8, i8, i8, i32, i32, i32 }
	%struct.GLDFramebufferAttachment = type { i32, i32, i32, i32, i32, i32 }
	%struct.GLDFramebufferData = type { [6 x %struct.GLDFramebufferAttachment], [4 x i16], i16, i16, i16, i16, i32 }
	%struct.GLDFramebufferRec = type { ptr, ptr, %struct.GLDPixelFormat }
	%struct.GLDHintMode = type { i16, i16, i16, i16, i16, i16, i16, i16, i16, i16 }
	%struct.GLDHistogram = type { ptr, i32, i16, i8, i8 }
	%struct.GLDImagingColorScale = type { { float, float }, { float, float }, { float, float }, { float, float } }
	%struct.GLDImagingSubset = type { %struct.GLDConvolution, %struct.GLDConvolution, %struct.GLDConvolution, %struct.GLDColorMatrix, %struct.GLDMinmax, %struct.GLDHistogram, %struct.GLDImagingColorScale, %struct.GLDImagingColorScale, %struct.GLDImagingColorScale, %struct.GLDImagingColorScale, i32 }
	%struct.GLDLight = type { %struct.GLTColor4, %struct.GLTColor4, %struct.GLTColor4, %struct.GLTColor4, %struct.GLTCoord3, float, float, float, float, float, %struct.GLTCoord3, float, float, float, float, float }
	%struct.GLDLightModel = type { %struct.GLTColor4, [8 x %struct.GLDLight], [2 x %struct.GLDMaterial], i32, i16, i16, i16, i8, i8, i8, i8, i8, i8 }
	%struct.GLDLightProduct = type { %struct.GLTColor4, %struct.GLTColor4, %struct.GLTColor4 }
	%struct.GLDLineMode = type { float, i32, i16, i16, i8, i8, i8, i8 }
	%struct.GLDLogicOp = type { i16, i8, i8 }
	%struct.GLDMaskMode = type { i32, [3 x i32], i8, i8, i8, i8, i8, i8, i8, i8 }
	%struct.GLDMaterial = type { %struct.GLTColor4, %struct.GLTColor4, %struct.GLTColor4, %struct.GLTColor4, float, float, float, float, [8 x %struct.GLDLightProduct], %struct.GLTColor4, [6 x i32], [2 x i32] }
	%struct.GLDMinmax = type { ptr, i16, i8, i8 }
	%struct.GLDMinmaxTable = type { %struct.GLTColor4, %struct.GLTColor4 }
	%struct.GLDMipmaplevel = type { [4 x i32], [4 x float], [4 x i32], [4 x i32], [4 x float], [4 x i32], [3 x i32], i32, ptr, ptr, ptr, i32, i32, ptr, i16, i16, i16, i16 }
	%struct.GLDMultisample = type { float, i8, i8, i8, i8, i8, i8, i8, i8 }
	%struct.GLDPipelineProgramData = type { i16, i16, i32, ptr, i64, ptr, ptr, i32 }
	%struct.GLDPipelineProgramRec = type { ptr, ptr, ptr, ptr, i32, i32, i32 }
	%struct.GLDPipelineProgramState = type { i8, i8, i8, i8, ptr }
	%struct.GLDPixelFormat = type { i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }
	%struct.GLDPixelMap = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.GLDPixelMode = type { float, float, %struct.GLDPixelStore, %struct.GLDPixelTransfer, %struct.GLDPixelMap, %struct.GLDImagingSubset, i32, i32 }
	%struct.GLDPixelPack = type { i32, i32, i32, i32, i32, i32, i32, i32, i8, i8, i8, i8 }
	%struct.GLDPixelStore = type { %struct.GLDPixelPack, %struct.GLDPixelPack }
	%struct.GLDPixelTransfer = type { float, float, float, float, float, float, float, float, float, float, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float }
	%struct.GLDPluginFramebufferData = type { [6 x ptr], i32, i32 }
	%struct.GLDPluginProgramData = type { [3 x ptr], ptr, i32 }
	%struct.GLDPluginState = type { [16 x [5 x ptr]], [3 x ptr], [16 x ptr], [3 x ptr], ptr, ptr, [16 x ptr], ptr, ptr }
	%struct.GLDPointMode = type { float, float, float, float, %struct.GLTCoord3, float, i8, i8, i8, i8, i16, i16, i32, i16, i16 }
	%struct.GLDPolygonMode = type { [128 x i8], float, float, i16, i16, i16, i16, i8, i8, i8, i8, i8, i8, i8, i8 }
	%struct.GLDProgramData = type { i32, [16 x i32], i32, i32, i32, i32 }
	%struct.GLDProgramRec = type { ptr, ptr, i32 }
	%struct.GLDQueryRec = type { i32, i32, ptr }
	%struct.GLDRect = type { i32, i32, i32, i32, i32, i32 }
	%struct.GLDRegisterCombiners = type { i8, i8, i8, i8, i32, [2 x %struct.GLTColor4], [8 x %struct.GLDRegisterCombinersPerStageState], %struct.GLDRegisterCombinersFinalStageState }
	%struct.GLDRegisterCombinersFinalStageState = type { i8, i8, i8, i8, [7 x %struct.GLDRegisterCombinersPerVariableState] }
	%struct.GLDRegisterCombinersPerPortionState = type { [4 x %struct.GLDRegisterCombinersPerVariableState], i8, i8, i8, i8, i16, i16, i16, i16, i16, i16 }
	%struct.GLDRegisterCombinersPerStageState = type { [2 x %struct.GLDRegisterCombinersPerPortionState], [2 x %struct.GLTColor4] }
	%struct.GLDRegisterCombinersPerVariableState = type { i16, i16, i16, i16 }
	%struct.GLDScissorTest = type { %struct.GLTFixedColor4, i8, i8, i8, i8 }
	%struct.GLDShaderSourceData = type { i32, i32, ptr, ptr, i32, i32, ptr, ptr, ptr }
	%struct.GLDSharedRec = type opaque
	%struct.GLDState = type { i16, i16, i32, i32, i32, [256 x %struct.GLTColor4], [128 x %struct.GLTColor4], %struct.GLDViewport, %struct.GLDTransform, %struct.GLDLightModel, ptr, i32, i32, i32, %struct.GLDAlphaTest, %struct.GLDBlendMode, %struct.GLDClearColor, %struct.GLDColorBuffer, %struct.GLDDepthTest, %struct.GLDArrayRange, %struct.GLDFogMode, %struct.GLDHintMode, %struct.GLDLineMode, %struct.GLDLogicOp, %struct.GLDMaskMode, %struct.GLDPixelMode, %struct.GLDPointMode, %struct.GLDPolygonMode, %struct.GLDScissorTest, i32, %struct.GLDStencilTest, [16 x %struct.GLDTextureMode], %struct.GLDArrayRange, [8 x %struct.GLDTextureCoordGen], %struct.GLDClipPlane, %struct.GLDMultisample, %struct.GLDRegisterCombiners, %struct.GLDArrayRange, %struct.GLDArrayRange, [3 x %struct.GLDPipelineProgramState], %struct.GLDTransformFeedback }
	%struct.GLDStateProgramRec = type { ptr, ptr }
	%struct.GLDStencilTest = type { [3 x { i32, i32, i16, i16, i16, i16 }], i32, [4 x i8] }
	%struct.GLDStippleData = type { i32, i16, i16, [32 x [32 x i8]] }
	%struct.GLDTextureCoordGen = type { { i16, i16, %struct.GLTColor4, %struct.GLTColor4 }, { i16, i16, %struct.GLTColor4, %struct.GLTColor4 }, { i16, i16, %struct.GLTColor4, %struct.GLTColor4 }, { i16, i16, %struct.GLTColor4, %struct.GLTColor4 }, i8, i8, i8, i8 }
	%struct.GLDTextureGeomState = type { i16, i16, i16, i16, i16, i8, i8, i16, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, [6 x i16], [6 x i16] }
	%struct.GLDTextureLevel = type { i32, i32, i16, i16, i16, i8, i8, i16, i16, i16, i16, ptr }
	%struct.GLDTextureMachine = type { [8 x ptr], ptr, i8, i8, i8, i8 }
	%struct.GLDTextureMode = type { %struct.GLTColor4, i32, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, float, float, float, i16, i16, i16, i16, i16, i16, [4 x i16], i8, i8, i8, i8, [3 x float], [4 x float], float, float }
	%struct.GLDTextureParamState = type { i16, i16, i16, i16, i16, i16, %struct.GLTColor4, float, float, float, float, i16, i16, i16, i16, float, i16, i8, i8, i32, ptr }
	%struct.GLDTextureRec = type { ptr, i32, [2 x float], float, i32, float, float, float, float, float, float, ptr, ptr, i32, i32, i32, i32, i32, i32, %struct.GLDTextureParamState, i32, [2 x %struct._PPStreamToken] }
	%struct.GLDTextureState = type { i16, i16, i16, float, i32, i16, ptr, i8, i8, i8, i8, %struct.GLDTextureParamState, %struct.GLDTextureGeomState, %struct.GLDTextureLevel, [6 x [15 x %struct.GLDTextureLevel]] }
	%struct.GLDTransform = type { [24 x [16 x float]], [24 x [16 x float]], [16 x float], float, float, float, float, i32, float, i16, i16, i8, i8, i8, i8 }
	%struct.GLDTransformFeedback = type { i8, i8, i8, [16 x i32], [16 x i32] }
	%struct.GLDVertex = type { %struct.GLTColor4, %struct.GLTColor4, %struct.GLTColor4, %struct.GLTColor4, %struct.GLTColor4, %struct.GLTCoord3, float, %struct.GLTColor4, float, float, float, i8, i8, i8, i8, [4 x float], [2 x ptr], i32, i32, [8 x %struct.GLTColor4] }
	%struct.GLDVertexArrayRec = type opaque
	%struct.GLDViewport = type { float, float, float, float, float, float, float, float, double, double, i32, i32, i32, i32, float, float, float, float }
	%struct.GLGColorTable = type { i32, i32, i32, ptr }
	%struct.GLGOperation = type { ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, float, %struct.GLGColorTable, %struct.GLGColorTable, %struct.GLGColorTable }
	%struct.GLGProcessor = type { ptr, ptr, ptr }
	%struct.GLISWRSurface = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr, [4 x ptr], i32 }
	%struct.GLIWindow = type { i32, i32, i32 }
	%struct.GLSBuffer = type { ptr }
	%struct.GLSDrawable = type { ptr }
	%struct.GLSWindowRec = type { %struct.GLTDimensions, %struct.GLTDimensions, i32, i32, %struct.GLSDrawable, [2 x ptr], ptr, ptr, ptr, [4 x ptr], i32, i32, i32, i32, [4 x i32], i16, i16, i16, %struct.GLIWindow, i32, i32, ptr, ptr }
	%struct.GLTColor4 = type { float, float, float, float }
	%struct.GLTCoord3 = type { float, float, float }
	%struct.GLTDimensions = type { i32, i32 }
	%struct.GLTFixedColor4 = type { i32, i32, i32, i32 }
	%struct.GLVMFPContext = type { float, i32, i32, i32 }
	%struct.GLVMFragmentAttribRec = type { <4 x float>, <4 x float>, <4 x float>, <4 x float>, [8 x <4 x float>] }
	%struct.GLVMTextures = type { [8 x ptr] }
	%struct._GLGFunctionKey = type opaque
	%struct._GLVMConstants = type opaque
	%struct._GLVMFunction = type opaque
	%struct._PPStreamToken = type { { i16, i8, i8, i32 } }

define void @gldLLVMVecPointRender(ptr %ctx) {
entry:
	%tmp.uip = getelementptr %struct.GLDContextRec, ptr %ctx, i32 0, i32 22		; <ptr> [#uses=1]
	%tmp = load i32, ptr %tmp.uip		; <i32> [#uses=3]
	%tmp91 = lshr i32 %tmp, 5		; <i32> [#uses=1]
	%tmp92 = trunc i32 %tmp91 to i1		; <i1> [#uses=1]
	br i1 %tmp92, label %cond_true93, label %cond_next116
cond_true93:		; preds = %entry
	%tmp.upgrd.1 = getelementptr %struct.GLDContextRec, ptr %ctx, i32 0, i32 31, i32 14		; <ptr> [#uses=1]
	%tmp95 = load i32, ptr %tmp.upgrd.1		; <i32> [#uses=1]
	%tmp95.upgrd.2 = sitofp i32 %tmp95 to float		; <float> [#uses=1]
	%tmp108 = fmul float undef, %tmp95.upgrd.2		; <float> [#uses=1]
	br label %cond_next116
cond_next116:		; preds = %cond_true93, %entry
	%point_size.2 = phi float [ %tmp108, %cond_true93 ], [ undef, %entry ]		; <float> [#uses=2]
	%tmp457 = fcmp olt float %point_size.2, 1.000000e+00		; <i1> [#uses=1]
	%tmp460 = lshr i32 %tmp, 6		; <i32> [#uses=1]
	%tmp461 = trunc i32 %tmp460 to i1		; <i1> [#uses=1]
	br i1 %tmp457, label %cond_true458, label %cond_next484
cond_true458:		; preds = %cond_next116
	br i1 %tmp461, label %cond_true462, label %cond_next487
cond_true462:		; preds = %cond_true458
	%tmp26 = bitcast i32 %tmp to i32		; <i32> [#uses=1]
	%tmp465 = and i32 %tmp26, 128		; <i32> [#uses=1]
	%tmp466 = icmp eq i32 %tmp465, 0		; <i1> [#uses=1]
	br i1 %tmp466, label %cond_true467, label %cond_next487
cond_true467:		; preds = %cond_true462
	ret void
cond_next484:		; preds = %cond_next116
	%tmp486 = fmul float %point_size.2, 5.000000e-01		; <float> [#uses=1]
	br label %cond_next487
cond_next487:		; preds = %cond_next484, %cond_true462, %cond_true458
	%radius.0 = phi float [ %tmp486, %cond_next484 ], [ 5.000000e-01, %cond_true458 ], [ 5.000000e-01, %cond_true462 ]		; <float> [#uses=2]
	%tmp494 = insertelement <4 x float> zeroinitializer, float %radius.0, i32 2		; <<4 x float>> [#uses=1]
	%tmp495 = insertelement <4 x float> %tmp494, float %radius.0, i32 3		; <<4 x float>> [#uses=0]
	ret void
}
