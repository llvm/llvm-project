; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin9.5 -mattr=+sse41 -relocation-model=pic

	%struct.XXActiveTextureTargets = type { i64, i64, i64, i64, i64, i64 }
	%struct.XXAlphaTest = type { float, i16, i8, i8 }
	%struct.XXArrayRange = type { i8, i8, i8, i8 }
	%struct.XXBlendMode = type { i16, i16, i16, i16, %struct.ZZIColor4, i16, i16, i8, i8, i8, i8 }
	%struct.XXBBRec = type opaque
	%struct.XXBBstate = type { %struct.ZZGTransformKey, %struct.ZZGTransformKey, %struct.XXProgramLimits, %struct.XXProgramLimits, i8, i8, i8, i8, %struct.ZZSBB, %struct.ZZSBB, [4 x %struct.ZZSBB], %struct.ZZSBB, %struct.ZZSBB, %struct.ZZSBB, [8 x %struct.ZZSBB], %struct.ZZSBB }
	%struct.XXClearColor = type { double, %struct.ZZIColor4, %struct.ZZIColor4, float, i32 }
	%struct.XXClipPlane = type { i32, [6 x %struct.ZZIColor4] }
	%struct.XXColorBB = type { i16, i8, i8, [8 x i16], i8, i8, i8, i8 }
	%struct.XXColorMatrix = type { [16 x float]*, %struct.XXImagingColorScale }
	%struct.XXConfig = type { i32, float, %struct.ZZGTransformKey, %struct.ZZGTransformKey, i8, i8, i8, i8, i8, i8, i16, i32, i32, i32, %struct.XXPixelFormatInfo, %struct.XXPointLineLimits, %struct.XXPointLineLimits, %struct.XXRenderFeatures, i32, i32, i32, i32, i32, i32, i32, i32, i32, %struct.XXTextureLimits, [3 x %struct.XXPipelineProgramLimits], %struct.XXFragmentProgramLimits, %struct.XXVertexProgramLimits, %struct.XXGeometryShaderLimits, %struct.XXProgramLimits, %struct.XXGeometryShaderLimits, %struct.XXVertexDescriptor*, %struct.XXVertexDescriptor*, [3 x i32], [4 x i32], [0 x i32] }
	%struct.XXContextRec = type { float, float, float, float, float, float, float, float, %struct.ZZIColor4, %struct.ZZIColor4, %struct.YYFPContext, [16 x [2 x %struct.PPStreamToken]], %struct.ZZGProcessor, %struct._YYConstants*, void (%struct.XXContextRec*, i32, i32, %struct.YYFragmentAttrib*, %struct.YYFragmentAttrib*, i32)*, %struct._YYFunction*, %struct.PPStreamToken*, void (%struct.XXContextRec*, %struct.XXVertex*)*, void (%struct.XXContextRec*, %struct.XXVertex*, %struct.XXVertex*)*, void (%struct.XXContextRec*, %struct.XXVertex*, %struct.XXVertex*, %struct.XXVertex*)*, %struct._YYFunction*, %struct._YYFunction*, %struct._YYFunction*, [4 x i32], [3 x i32], [3 x i32], float, float, float, %struct.PPStreamToken, i32, %struct.ZZSDrawable, %struct.XXFramebufferRec*, %struct.XXFramebufferRec*, %struct.XXRect, %struct.XXFormat, %struct.XXFormat, %struct.XXFormat, %struct.XXConfig*, %struct.XXBBstate, %struct.XXBBstate, %struct.XXSharedRec*, %struct.XXState*, %struct.XXPluginState*, %struct.XXVertex*, %struct.YYFragmentAttrib*, %struct.YYFragmentAttrib*, %struct.YYFragmentAttrib*, %struct.XXProgramRec*, %struct.XXPipelineProgramRec*, %struct.YYTextures, %struct.XXStippleData, i8, i16, i8, i32, i32, i32, %struct.XXQueryRec*, %struct.XXQueryRec*, %struct.XXFallback, { void (i8*, i8*, i32, i8*)* } }
	%struct.XXConvolution = type { %struct.ZZIColor4, %struct.XXImagingColorScale, i16, i16, [0 x i32], float*, i32, i32 }
	%struct.XXCurrent16A = type { [8 x %struct.ZZIColor4], [16 x %struct.ZZIColor4], %struct.ZZIColor4, %struct.XXPointLineLimits, float, %struct.XXPointLineLimits, float, [4 x float], %struct.XXPointLineLimits, float, float, float, float, i8, i8, i8, i8 }
	%struct.XXDepthTest = type { i16, i16, i8, i8, i8, i8, double, double }
	%struct.XXDrawableWindow = type { i32, i32, i32 }
	%struct.XXFallback = type { float*, %struct.XXRenderDispatch*, %struct.XXConfig*, i8*, i8*, i32, i32 }
	%struct.XXFenceRec = type opaque
	%struct.XXFixedFunction = type { %struct.PPStreamToken* }
	%struct.XXFogMode = type { %struct.ZZIColor4, float, float, float, float, float, i16, i16, i16, i8, i8 }
	%struct.XXFormat = type { i32, i32, i32, i32, i32, i32, i32, i32, i8, i8, i8, i8, i32, i32, i32 }
	%struct.XXFragmentProgramLimits = type { i32, i32, i32, i16, i16, i32, i32 }
	%struct.XXFramebufferAttachment = type { i16, i16, i32, i32, i32 }
	%struct.XXFramebufferData = type { [10 x %struct.XXFramebufferAttachment], [8 x i16], i16, i16, i16, i8, i8, i32, i32 }
	%struct.XXFramebufferRec = type { %struct.XXFramebufferData*, %struct.XXPluginFramebufferData*, %struct.XXFormat, i8, i8, i8, i8 }
	%struct.XXGeometryShaderLimits = type { i32, i32, i32, i32, i32 }
	%struct.XXHintMode = type { i16, i16, i16, i16, i16, i16, i16, i16, i16, i16 }
	%struct.XXHistogram = type { %struct.XXProgramLimits*, i32, i16, i8, i8 }
	%struct.XXImagingColorScale = type { %struct.ZZTCoord2, %struct.ZZTCoord2, %struct.ZZTCoord2, %struct.ZZTCoord2 }
	%struct.XXImagingSubset = type { %struct.XXConvolution, %struct.XXConvolution, %struct.XXConvolution, %struct.XXColorMatrix, %struct.XXMinmax, %struct.XXHistogram, %struct.XXImagingColorScale, %struct.XXImagingColorScale, %struct.XXImagingColorScale, %struct.XXImagingColorScale, i32, [0 x i32] }
	%struct.XXLight = type { %struct.ZZIColor4, %struct.ZZIColor4, %struct.ZZIColor4, %struct.ZZIColor4, %struct.XXPointLineLimits, float, float, float, float, float, %struct.XXPointLineLimits, float, %struct.XXPointLineLimits, float, %struct.XXPointLineLimits, float, float, float, float, float }
	%struct.XXLightModel = type { %struct.ZZIColor4, [8 x %struct.XXLight], [2 x %struct.XXMaterial], i32, i16, i16, i16, i8, i8, i8, i8, i8, i8 }
	%struct.XXLightProduct = type { %struct.ZZIColor4, %struct.ZZIColor4, %struct.ZZIColor4 }
	%struct.XXLineMode = type { float, i32, i16, i16, i8, i8, i8, i8 }
	%struct.XXLogicOp = type { i16, i8, i8 }
	%struct.XXMaskMode = type { i32, [3 x i32], i8, i8, i8, i8, i8, i8, i8, i8 }
	%struct.XXMaterial = type { %struct.ZZIColor4, %struct.ZZIColor4, %struct.ZZIColor4, %struct.ZZIColor4, float, float, float, float, [8 x %struct.XXLightProduct], %struct.ZZIColor4, [8 x i32] }
	%struct.XXMinmax = type { %struct.XXMinmaxTable*, i16, i8, i8, [0 x i32] }
	%struct.XXMinmaxTable = type { %struct.ZZIColor4, %struct.ZZIColor4 }
	%struct.XXMipmaplevel = type { [4 x i32], [4 x i32], [4 x float], [4 x i32], i32, i32, float*, i8*, i16, i16, i16, i16, [2 x float] }
	%struct.XXMultisample = type { float, i8, i8, i8, i8, i8, i8, i8, i8 }
	%struct.XXPipelineProgramData = type { i16, i8, i8, i32, %struct.PPStreamToken*, i64, %struct.ZZIColor4*, i32, [0 x i32] }
	%struct.XXPipelineProgramLimits = type { i32, i16, i16, i32, i16, i16, i32, i32 }
	%struct.XXPipelineProgramRec = type { %struct.XXPipelineProgramData*, %struct.PPStreamToken*, %struct.XXContextRec*, { %struct._YYFunction*, \2, \2, [20 x i32], [64 x i32], i32, i32, i32 }*, i32, i32 }
	%struct.XXPipelineProgramState = type { i8, i8, i8, i8, [0 x i32], %struct.ZZIColor4* }
	%struct.XXPixelFormatInfo = type { i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }
	%struct.XXPixelMap = type { i32*, float*, float*, float*, float*, float*, float*, float*, float*, i32*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.XXPixelMode = type { float, float, %struct.XXPixelStore, %struct.XXPixelTransfer, %struct.XXPixelMap, %struct.XXImagingSubset, i32, i32 }
	%struct.XXPixelPack = type { i32, i32, i32, i32, i32, i32, i32, i32, i8, i8, i8, i8 }
	%struct.XXPixelStore = type { %struct.XXPixelPack, %struct.XXPixelPack }
	%struct.XXPixelTransfer = type { float, float, float, float, float, float, float, float, float, float, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float }
	%struct.XXPluginFramebufferData = type { [10 x %struct.XXTextureRec*], i8, i8, i8, i8 }
	%struct.XXPluginProgramData = type { [3 x %struct.XXPipelineProgramRec*], %struct.XXBBRec**, i32, [0 x i32] }
	%struct.XXPluginState = type { [16 x [5 x %struct.XXTextureRec*]], [3 x %struct.XXTextureRec*], [3 x %struct.XXPipelineProgramRec*], [3 x %struct.XXPipelineProgramRec*], %struct.XXProgramRec*, %struct.XXVertexArrayRec*, [16 x %struct.XXBBRec*], %struct.XXFramebufferRec*, %struct.XXFramebufferRec* }
	%struct.XXPointLineLimits = type { float, float, float }
	%struct.XXPointMode = type { float, float, float, float, %struct.XXPointLineLimits, float, i8, i8, i8, i8, i16, i16, i32, i16, i16 }
	%struct.XXPolygonMode = type { [128 x i8], float, float, i16, i16, i16, i16, i8, i8, i8, i8, i8, i8, i8, i8 }
	%struct.XXProgramData = type { i32, i32, i32, i32, %struct.PPStreamToken*, i32*, i32, i32, i32, i32, i8, i8, i8, i8, [0 x i32] }
	%struct.XXProgramLimits = type { i32, i32, i32, i32 }
	%struct.XXProgramRec = type { %struct.XXProgramData*, %struct.XXPluginProgramData*, %struct.ZZIColor4**, i32 }
	%struct.XXQueryRec = type { i32, i32, %struct.XXQueryRec* }
	%struct.XXRect = type { i32, i32, i32, i32, i32, i32 }
	%struct.XXRegisterCombiners = type { i8, i8, i8, i8, i32, [2 x %struct.ZZIColor4], [8 x %struct.XXRegisterCombinersPerStageState], %struct.XXRegisterCombinersFinalStageState }
	%struct.XXRegisterCombinersFinalStageState = type { i8, i8, i8, i8, [7 x %struct.XXRegisterCombinersPerVariableState] }
	%struct.XXRegisterCombinersPerPortionState = type { [4 x %struct.XXRegisterCombinersPerVariableState], i8, i8, i8, i8, i16, i16, i16, i16, i16, i16 }
	%struct.XXRegisterCombinersPerStageState = type { [2 x %struct.XXRegisterCombinersPerPortionState], [2 x %struct.ZZIColor4] }
	%struct.XXRegisterCombinersPerVariableState = type { i16, i16, i16, i16 }
	%struct.XXRenderDispatch = type { void (%struct.XXContextRec*, i32, float)*, void (%struct.XXContextRec*, i32)*, i32 (%struct.XXContextRec*, i32, i32, i32, i32, i32, i32, i8*, i32, %struct.XXBBRec*)*, i32 (%struct.XXContextRec*, %struct.XXVertex*, i32, i32, i32, i32, i8*, i32, %struct.XXBBRec*)*, void (%struct.XXContextRec*, %struct.XXVertex*, i32, i32, i32, i32, i32)*, void (%struct.XXContextRec*, %struct.XXVertex*, i32, i32, float, float, i8*, i32)*, void (%struct.XXContextRec*, %struct.XXVertex*, i32, i32)*, void (%struct.XXContextRec*, %struct.XXVertex*, i32, i32)*, void (%struct.XXContextRec*, %struct.XXVertex*, i32, i32)*, void (%struct.XXContextRec*, %struct.XXVertex*, i32, i32)*, void (%struct.XXContextRec*, %struct.XXVertex*, i32, i32)*, void (%struct.XXContextRec*, %struct.XXVertex*, i32, i32)*, void (%struct.XXContextRec*, %struct.XXVertex*, %struct.XXVertex*, i32, i32)*, void (%struct.XXContextRec*, %struct.XXVertex*, i32, i32)*, void (%struct.XXContextRec*, %struct.XXVertex*, i32, i32)*, void (%struct.XXContextRec*, %struct.XXVertex*, i32, i32)*, void (%struct.XXContextRec*, %struct.XXVertex*, i32, i32)*, void (%struct.XXContextRec*, %struct.XXVertex*, i32, i32)*, void (%struct.XXContextRec*, %struct.XXVertex*, i32, i32)*, void (%struct.XXContextRec*, %struct.XXVertex*, i32, i32)*, void (%struct.XXContextRec*, %struct.XXVertex**, i32)*, void (%struct.XXContextRec*, %struct.XXVertex**, i32, i32)*, void (%struct.XXContextRec*, %struct.XXVertex**, i32, i32)*, i8* (%struct.XXContextRec*, i32, i32*)*, void (%struct.XXContextRec*, i32, i32, i32)*, i8* (%struct.XXContextRec*, i32, i32, i32, i32, i32)*, void (%struct.XXContextRec*, i32, i32, i32, i32, i32, i8*)*, void (%struct.XXContextRec*)*, void (%struct.XXContextRec*)*, void (%struct.XXContextRec*)*, void (%struct.XXContextRec*, %struct.XXFenceRec*)*, void (%struct.XXContextRec*, i32, %struct.XXQueryRec*)*, void (%struct.XXContextRec*, %struct.XXQueryRec*)*, i32 (%struct.XXContextRec*, i32, i32, i32, i32, i32, i8*, %struct.ZZIColor4*, %struct.XXCurrent16A*)*, i32 (%struct.XXContextRec*, %struct.XXTextureRec*, i32, i32, i32, i32, i32, i32, i32, i32, i32)*, i32 (%struct.XXContextRec*, %struct.XXTextureRec*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8*, i32, %struct.XXBBRec*)*, i32 (%struct.XXContextRec*, %struct.XXTextureRec*, i32)*, i32 (%struct.XXContextRec*, %struct.XXBBRec*, i32, i32, i8*)*, void (%struct.XXContextRec*, i32)*, void (%struct.XXContextRec*)*, void (%struct.XXContextRec*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)*, i32 (%struct.XXContextRec*, %struct.XXQueryRec*)*, void (%struct.XXContextRec*)* }
	%struct.XXRenderFeatures = type { i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }
	%struct.XXSWRSurfaceRec = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8*, i8*, i8*, [4 x i8*], i32 }
	%struct.XXScissorTest = type { %struct.XXProgramLimits, i8, i8, i8, i8 }
	%struct.XXSharedData = type {  }
	%struct.XXSharedRec = type { %struct.__ZZarrayelementDrawInfoListType, %struct.XXSharedData*, i32, i8, i8, i8, i8 }
	%struct.XXState = type <{ i16, i16, i16, i16, i32, i32, [256 x %struct.ZZIColor4], [128 x %struct.ZZIColor4], %struct.XXViewport, %struct.XXTransform, %struct.XXLightModel, %struct.XXActiveTextureTargets, %struct.XXAlphaTest, %struct.XXBlendMode, %struct.XXClearColor, %struct.XXColorBB, %struct.XXDepthTest, %struct.XXArrayRange, %struct.XXFogMode, %struct.XXHintMode, %struct.XXLineMode, %struct.XXLogicOp, %struct.XXMaskMode, %struct.XXPixelMode, %struct.XXPointMode, %struct.XXPolygonMode, %struct.XXScissorTest, i32, %struct.XXStencilTest, [8 x %struct.XXTextureMode], [16 x %struct.XXTextureImageMode], %struct.XXArrayRange, [8 x %struct.XXTextureCoordGen], %struct.XXClipPlane, %struct.XXMultisample, %struct.XXRegisterCombiners, %struct.XXArrayRange, %struct.XXArrayRange, [3 x %struct.XXPipelineProgramState], %struct.XXArrayRange, %struct.XXTransformFeedback, i32*, %struct.XXFixedFunction, [1 x i32] }>
	%struct.XXStencilTest = type { [3 x { i32, i32, i16, i16, i16, i16 }], i32, [4 x i8] }
	%struct.XXStippleData = type { i32, i16, i16, [32 x [32 x i8]] }
	%struct.XXTextureCoordGen = type { { i16, i16, %struct.ZZIColor4, %struct.ZZIColor4 }, { i16, i16, %struct.ZZIColor4, %struct.ZZIColor4 }, { i16, i16, %struct.ZZIColor4, %struct.ZZIColor4 }, { i16, i16, %struct.ZZIColor4, %struct.ZZIColor4 }, i8, i8, i8, i8 }
	%struct.XXTextureGeomState = type { i16, i16, i16, i16, i16, i8, i8, i8, i8, i16, i16, i16, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, [6 x i16], [6 x i16] }
	%struct.XXTextureImageMode = type { float }
	%struct.XXTextureLevel = type { i32, i32, i16, i16, i16, i8, i8, i16, i16, i16, i16, i8* }
	%struct.XXTextureLimits = type { float, float, i16, i16, i16, i16, i16, i16, i16, i16, i16, i8, i8, [16 x i16], i32 }
	%struct.XXTextureMode = type { %struct.ZZIColor4, i32, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, float, float, i16, i16, i16, i16, i16, i16, [4 x i16], i8, i8, i8, i8, [3 x float], [4 x float], float, float }
	%struct.XXTextureParamState = type { i16, i16, i16, i16, i16, i16, %struct.ZZIColor4, float, float, float, float, i16, i16, i16, i16, float, i16, i8, i8, i32, i8* }
	%struct.XXTextureRec = type { [4 x float], %struct.XXTextureState*, %struct.XXMipmaplevel*, %struct.XXMipmaplevel*, float, float, float, float, i8, i8, i8, i8, i16, i16, i16, i16, i32, float, [2 x %struct.PPStreamToken] }
	%struct.XXTextureState = type { i16, i8, i8, i16, i16, float, i32, %struct.XXSWRSurfaceRec*, %struct.XXTextureParamState, %struct.XXTextureGeomState, i16, i16, i8*, %struct.XXTextureLevel, [1 x [15 x %struct.XXTextureLevel]] }
	%struct.XXTransform = type <{ [24 x [16 x float]], [24 x [16 x float]], [16 x float], float, float, float, float, float, i8, i8, i8, i8, i32, i32, i32, i16, i16, i8, i8, i8, i8, i32 }>
	%struct.XXTransformFeedback = type { i8, i8, i8, i8, [0 x i32], [16 x i32], [16 x i32] }
	%struct.XXVertex = type { %struct.ZZIColor4, %struct.ZZIColor4, %struct.ZZIColor4, %struct.ZZIColor4, %struct.ZZIColor4, %struct.XXPointLineLimits, float, %struct.ZZIColor4, float, i8, i8, i8, i8, float, float, i32, i32, i32, i32, [4 x float], [2 x %struct.XXMaterial*], [2 x i32], [8 x %struct.ZZIColor4] }
	%struct.XXVertexArrayRec = type opaque
	%struct.XXVertexDescriptor = type { i8, i8, i8, i8, [0 x i32] }
	%struct.XXVertexProgramLimits = type { i16, i16, i32, i32 }
	%struct.XXViewport = type { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, double, double, i32, i32, i32, i32, float, float, float, float }
	%struct.ZZGColorTable = type { i32, i32, i32, i8* }
	%struct.ZZGOperation = type { i8*, i8*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, float, %struct.ZZGColorTable, %struct.ZZGColorTable, %struct.ZZGColorTable }
	%struct.ZZGProcessor = type { void (%struct.XXPixelMode*, %struct.ZZGOperation*, %struct._ZZGProcessorData*, %union._ZZGFunctionKey*)*, %struct._YYFunction*, %union._ZZGFunctionKey*, %struct._ZZGProcessorData* }
	%struct.ZZGTransformKey = type { i32, i32 }
	%struct.ZZIColor4 = type { float, float, float, float }
	%struct.ZZSBB = type { i8* }
	%struct.ZZSDrawable = type { %struct.ZZSWindowRec* }
	%struct.ZZSWindowRec = type { %struct.ZZGTransformKey, %struct.ZZGTransformKey, i32, i32, %struct.ZZSDrawable, i8*, i8*, i8*, i8*, i8*, [4 x i8*], i32, i16, i16, i16, i16, i8, i8, i8, i8, i8, i8, i8, i8, %struct.XXDrawableWindow, i32, i32, i8*, i8* }
	%struct.ZZTCoord2 = type { float, float }
	%struct.YYFPContext = type { float, i32, i32, i32, float, [3 x float] }
	%struct.YYFragmentAttrib = type { <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, [8 x <4 x float>] }
	%struct.YYTextures = type { [16 x %struct.XXTextureRec*] }
	%struct.PPStreamToken = type { { i16, i16, i32 } }
	%struct._ZZGProcessorData = type { void (i8*, i8*, i32, i32, i32, i32, i32, i32, i32)*, void (i8*, i8*, i32, i32, i32, i32, i32, i32, i32)*, i8* (i32)*, void (i8*)* }
	%struct._YYConstants = type { <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, float, float, float, float, float, float, float, float, float, float, float, float, [256 x float], [4096 x i8], [8 x float], [48 x float], [128 x float], [528 x i8], { void (i8*, i8*, i32, i8*)*, float (float)*, float (float)*, float (float)*, i32 (float)* } }
	%struct._YYFunction = type opaque
	%struct.__ZZarrayelementDrawInfoListType = type { i32, [40 x i8] }
	%union._ZZGFunctionKey = type opaque
@llvm.used = appending global [1 x i8*] [ i8* bitcast (void (%struct.XXContextRec*, i32, i32, %struct.YYFragmentAttrib*, %struct.YYFragmentAttrib*, i32)* @t to i8*) ], section "llvm.metadata"		; <[1 x i8*]*> [#uses=0]

define void @t(%struct.XXContextRec* %ctx, i32 %x, i32 %y, %struct.YYFragmentAttrib* %start, %struct.YYFragmentAttrib* %deriv, i32 %num_frags) nounwind {
entry:
	%tmp7485.i.i.i = xor <4 x i32> zeroinitializer, < i32 -1, i32 -1, i32 -1, i32 -1 >		; <<4 x i32>> [#uses=1]
	%tmp8382.i.i.i = extractelement <4 x i32> zeroinitializer, i32 1		; <i32> [#uses=1]
	%tmp8383.i.i.i = extractelement <4 x i32> zeroinitializer, i32 2		; <i32> [#uses=2]
	%tmp8384.i.i.i = extractelement <4 x i32> zeroinitializer, i32 3		; <i32> [#uses=2]
	br label %bb7551.i.i.i

bb4426.i.i.i:		; preds = %bb7551.i.i.i
	%0 = getelementptr %struct.XXMipmaplevel* null, i32 %tmp8383.i.i.i, i32 3		; <[4 x i32]*> [#uses=1]
	%1 = bitcast [4 x i32]* %0 to <4 x i32>*		; <<4 x i32>*> [#uses=1]
	%2 = load <4 x i32>* %1, align 16		; <<4 x i32>> [#uses=1]
	%3 = getelementptr %struct.XXMipmaplevel* null, i32 %tmp8384.i.i.i, i32 3		; <[4 x i32]*> [#uses=1]
	%4 = bitcast [4 x i32]* %3 to <4 x i32>*		; <<4 x i32>*> [#uses=1]
	%5 = load <4 x i32>* %4, align 16		; <<4 x i32>> [#uses=1]
	%6 = shufflevector <4 x i32> %2, <4 x i32> %5, <4 x i32> < i32 0, i32 4, i32 1, i32 5 >		; <<4 x i32>> [#uses=1]
	%7 = bitcast <4 x i32> %6 to <2 x i64>		; <<2 x i64>> [#uses=1]
	%8 = shufflevector <2 x i64> zeroinitializer, <2 x i64> %7, <2 x i32> < i32 1, i32 3 >		; <<2 x i64>> [#uses=1]
	%9 = getelementptr %struct.XXMipmaplevel* null, i32 %tmp8382.i.i.i, i32 6		; <float**> [#uses=1]
	%10 = load float** %9, align 4		; <float*> [#uses=1]
	%11 = bitcast float* %10 to i8*		; <i8*> [#uses=1]
	%12 = getelementptr %struct.XXMipmaplevel* null, i32 %tmp8383.i.i.i, i32 6		; <float**> [#uses=1]
	%13 = load float** %12, align 4		; <float*> [#uses=1]
	%14 = bitcast float* %13 to i8*		; <i8*> [#uses=1]
	%15 = getelementptr %struct.XXMipmaplevel* null, i32 %tmp8384.i.i.i, i32 6		; <float**> [#uses=1]
	%16 = load float** %15, align 4		; <float*> [#uses=1]
	%17 = bitcast float* %16 to i8*		; <i8*> [#uses=1]
	%tmp7308.i.i.i = and <2 x i64> zeroinitializer, %8		; <<2 x i64>> [#uses=1]
	%18 = bitcast <2 x i64> %tmp7308.i.i.i to <4 x i32>		; <<4 x i32>> [#uses=1]
	%19 = mul <4 x i32> %18, zeroinitializer		; <<4 x i32>> [#uses=1]
	%20 = add <4 x i32> %19, zeroinitializer		; <<4 x i32>> [#uses=3]
	%21 = load i32* null, align 4		; <i32> [#uses=0]
	%22 = call <4 x float> @llvm.x86.sse2.cvtdq2ps(<4 x i32> zeroinitializer) nounwind readnone		; <<4 x float>> [#uses=1]
	%23 = mul <4 x float> %22, < float 0x3F70101020000000, float 0x3F70101020000000, float 0x3F70101020000000, float 0x3F70101020000000 >		; <<4 x float>> [#uses=1]
	%tmp2114.i119.i.i = extractelement <4 x i32> %20, i32 1		; <i32> [#uses=1]
	%24 = shl i32 %tmp2114.i119.i.i, 2		; <i32> [#uses=1]
	%25 = getelementptr i8* %11, i32 %24		; <i8*> [#uses=1]
	%26 = bitcast i8* %25 to i32*		; <i32*> [#uses=1]
	%27 = load i32* %26, align 4		; <i32> [#uses=1]
	%28 = or i32 %27, -16777216		; <i32> [#uses=1]
	%tmp1927.i120.i.i = insertelement <4 x i32> undef, i32 %28, i32 0		; <<4 x i32>> [#uses=1]
	%29 = bitcast <4 x i32> %tmp1927.i120.i.i to <16 x i8>		; <<16 x i8>> [#uses=1]
	%30 = shufflevector <16 x i8> %29, <16 x i8> < i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef >, <16 x i32> < i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23 >		; <<16 x i8>> [#uses=1]
	%31 = bitcast <16 x i8> %30 to <8 x i16>		; <<8 x i16>> [#uses=1]
	%32 = shufflevector <8 x i16> %31, <8 x i16> < i16 0, i16 0, i16 0, i16 0, i16 undef, i16 undef, i16 undef, i16 undef >, <8 x i32> < i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11 >		; <<8 x i16>> [#uses=1]
	%33 = bitcast <8 x i16> %32 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%34 = shufflevector <4 x i32> %33, <4 x i32> undef, <4 x i32> < i32 2, i32 1, i32 0, i32 3 >		; <<4 x i32>> [#uses=1]
	%35 = call <4 x float> @llvm.x86.sse2.cvtdq2ps(<4 x i32> %34) nounwind readnone		; <<4 x float>> [#uses=1]
	%36 = mul <4 x float> %35, < float 0x3F70101020000000, float 0x3F70101020000000, float 0x3F70101020000000, float 0x3F70101020000000 >		; <<4 x float>> [#uses=1]
	%tmp2113.i124.i.i = extractelement <4 x i32> %20, i32 2		; <i32> [#uses=1]
	%37 = shl i32 %tmp2113.i124.i.i, 2		; <i32> [#uses=1]
	%38 = getelementptr i8* %14, i32 %37		; <i8*> [#uses=1]
	%39 = bitcast i8* %38 to i32*		; <i32*> [#uses=1]
	%40 = load i32* %39, align 4		; <i32> [#uses=1]
	%41 = or i32 %40, -16777216		; <i32> [#uses=1]
	%tmp1963.i125.i.i = insertelement <4 x i32> undef, i32 %41, i32 0		; <<4 x i32>> [#uses=1]
	%42 = bitcast <4 x i32> %tmp1963.i125.i.i to <16 x i8>		; <<16 x i8>> [#uses=1]
	%43 = shufflevector <16 x i8> %42, <16 x i8> < i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef >, <16 x i32> < i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23 >		; <<16 x i8>> [#uses=1]
	%44 = bitcast <16 x i8> %43 to <8 x i16>		; <<8 x i16>> [#uses=1]
	%45 = shufflevector <8 x i16> %44, <8 x i16> < i16 0, i16 0, i16 0, i16 0, i16 undef, i16 undef, i16 undef, i16 undef >, <8 x i32> < i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11 >		; <<8 x i16>> [#uses=1]
	%46 = bitcast <8 x i16> %45 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%47 = shufflevector <4 x i32> %46, <4 x i32> undef, <4 x i32> < i32 2, i32 1, i32 0, i32 3 >		; <<4 x i32>> [#uses=1]
	%48 = call <4 x float> @llvm.x86.sse2.cvtdq2ps(<4 x i32> %47) nounwind readnone		; <<4 x float>> [#uses=1]
	%49 = mul <4 x float> %48, < float 0x3F70101020000000, float 0x3F70101020000000, float 0x3F70101020000000, float 0x3F70101020000000 >		; <<4 x float>> [#uses=1]
	%tmp2112.i129.i.i = extractelement <4 x i32> %20, i32 3		; <i32> [#uses=1]
	%50 = shl i32 %tmp2112.i129.i.i, 2		; <i32> [#uses=1]
	%51 = getelementptr i8* %17, i32 %50		; <i8*> [#uses=1]
	%52 = bitcast i8* %51 to i32*		; <i32*> [#uses=1]
	%53 = load i32* %52, align 4		; <i32> [#uses=1]
	%54 = or i32 %53, -16777216		; <i32> [#uses=1]
	%tmp1999.i130.i.i = insertelement <4 x i32> undef, i32 %54, i32 0		; <<4 x i32>> [#uses=1]
	%55 = bitcast <4 x i32> %tmp1999.i130.i.i to <16 x i8>		; <<16 x i8>> [#uses=1]
	%56 = shufflevector <16 x i8> %55, <16 x i8> < i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef >, <16 x i32> < i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23 >		; <<16 x i8>> [#uses=1]
	%57 = bitcast <16 x i8> %56 to <8 x i16>		; <<8 x i16>> [#uses=1]
	%58 = shufflevector <8 x i16> %57, <8 x i16> < i16 0, i16 0, i16 0, i16 0, i16 undef, i16 undef, i16 undef, i16 undef >, <8 x i32> < i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11 >		; <<8 x i16>> [#uses=1]
	%59 = bitcast <8 x i16> %58 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%60 = shufflevector <4 x i32> %59, <4 x i32> undef, <4 x i32> < i32 2, i32 1, i32 0, i32 3 >		; <<4 x i32>> [#uses=1]
	%61 = call <4 x float> @llvm.x86.sse2.cvtdq2ps(<4 x i32> %60) nounwind readnone		; <<4 x float>> [#uses=1]
	%62 = mul <4 x float> %61, < float 0x3F70101020000000, float 0x3F70101020000000, float 0x3F70101020000000, float 0x3F70101020000000 >		; <<4 x float>> [#uses=1]
	%63 = mul <4 x float> %23, zeroinitializer		; <<4 x float>> [#uses=1]
	%64 = add <4 x float> zeroinitializer, %63		; <<4 x float>> [#uses=1]
	%65 = mul <4 x float> %36, zeroinitializer		; <<4 x float>> [#uses=1]
	%66 = add <4 x float> zeroinitializer, %65		; <<4 x float>> [#uses=1]
	%67 = mul <4 x float> %49, zeroinitializer		; <<4 x float>> [#uses=1]
	%68 = add <4 x float> zeroinitializer, %67		; <<4 x float>> [#uses=1]
	%69 = mul <4 x float> %62, zeroinitializer		; <<4 x float>> [#uses=1]
	%70 = add <4 x float> zeroinitializer, %69		; <<4 x float>> [#uses=1]
	%tmp7452.i.i.i = bitcast <4 x float> %64 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp7454.i.i.i = and <4 x i32> %tmp7452.i.i.i, zeroinitializer		; <<4 x i32>> [#uses=1]
	%tmp7459.i.i.i = or <4 x i32> %tmp7454.i.i.i, zeroinitializer		; <<4 x i32>> [#uses=1]
	%tmp7460.i.i.i = bitcast <4 x i32> %tmp7459.i.i.i to <4 x float>		; <<4 x float>> [#uses=1]
	%tmp7468.i.i.i = bitcast <4 x float> %66 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp7470.i.i.i = and <4 x i32> %tmp7468.i.i.i, zeroinitializer		; <<4 x i32>> [#uses=1]
	%tmp7475.i.i.i = or <4 x i32> %tmp7470.i.i.i, zeroinitializer		; <<4 x i32>> [#uses=1]
	%tmp7476.i.i.i = bitcast <4 x i32> %tmp7475.i.i.i to <4 x float>		; <<4 x float>> [#uses=1]
	%tmp7479.i.i.i = bitcast <4 x float> %.279.1.i to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp7480.i.i.i = and <4 x i32> zeroinitializer, %tmp7479.i.i.i		; <<4 x i32>> [#uses=1]
	%tmp7484.i.i.i = bitcast <4 x float> %68 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp7486.i.i.i = and <4 x i32> %tmp7484.i.i.i, %tmp7485.i.i.i		; <<4 x i32>> [#uses=1]
	%tmp7491.i.i.i = or <4 x i32> %tmp7486.i.i.i, %tmp7480.i.i.i		; <<4 x i32>> [#uses=1]
	%tmp7492.i.i.i = bitcast <4 x i32> %tmp7491.i.i.i to <4 x float>		; <<4 x float>> [#uses=1]
	%tmp7495.i.i.i = bitcast <4 x float> %.380.1.i to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp7496.i.i.i = and <4 x i32> zeroinitializer, %tmp7495.i.i.i		; <<4 x i32>> [#uses=1]
	%tmp7500.i.i.i = bitcast <4 x float> %70 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp7502.i.i.i = and <4 x i32> %tmp7500.i.i.i, zeroinitializer		; <<4 x i32>> [#uses=1]
	%tmp7507.i.i.i = or <4 x i32> %tmp7502.i.i.i, %tmp7496.i.i.i		; <<4 x i32>> [#uses=1]
	%tmp7508.i.i.i = bitcast <4 x i32> %tmp7507.i.i.i to <4 x float>		; <<4 x float>> [#uses=1]
	%indvar.next.i.i.i = add i32 %aniso.0.i.i.i, 1		; <i32> [#uses=1]
	br label %bb7551.i.i.i

bb7551.i.i.i:		; preds = %bb4426.i.i.i, %entry
	%.077.1.i = phi <4 x float> [ undef, %entry ], [ %tmp7460.i.i.i, %bb4426.i.i.i ]		; <<4 x float>> [#uses=0]
	%.178.1.i = phi <4 x float> [ undef, %entry ], [ %tmp7476.i.i.i, %bb4426.i.i.i ]		; <<4 x float>> [#uses=0]
	%.279.1.i = phi <4 x float> [ undef, %entry ], [ %tmp7492.i.i.i, %bb4426.i.i.i ]		; <<4 x float>> [#uses=1]
	%.380.1.i = phi <4 x float> [ undef, %entry ], [ %tmp7508.i.i.i, %bb4426.i.i.i ]		; <<4 x float>> [#uses=1]
	%aniso.0.i.i.i = phi i32 [ 0, %entry ], [ %indvar.next.i.i.i, %bb4426.i.i.i ]		; <i32> [#uses=1]
	br i1 false, label %glvmInterpretFPTransformFour6.exit, label %bb4426.i.i.i

glvmInterpretFPTransformFour6.exit:		; preds = %bb7551.i.i.i
	unreachable
}

declare <4 x float> @llvm.x86.sse2.cvtdq2ps(<4 x i32>) nounwind readnone
