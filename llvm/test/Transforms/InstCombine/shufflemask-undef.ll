; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep {shufflevector.\*i32 8}

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9"
	%struct.ActiveTextureTargets = type { i64, i64, i64, i64, i64, i64 }
	%struct.AlphaTest = type { float, i16, i8, i8 }
	%struct.ArrayRange = type { i8, i8, i8, i8 }
	%struct.BlendMode = type { i16, i16, i16, i16, %struct.IColor4, i16, i16, i8, i8, i8, i8 }
	%struct.ClearColor = type { double, %struct.IColor4, %struct.IColor4, float, i32 }
	%struct.ClipPlane = type { i32, [6 x %struct.IColor4] }
	%struct.ColorBuffer = type { i16, i8, i8, [8 x i16], [0 x i32] }
	%struct.ColorMatrix = type { [16 x float]*, %struct.ImagingColorScale }
	%struct.Convolution = type { %struct.IColor4, %struct.ImagingColorScale, i16, i16, [0 x i32], float*, i32, i32 }
	%struct.DepthTest = type { i16, i16, i8, i8, i8, i8, double, double }
	%struct.FixedFunction = type { %struct.PPStreamToken* }
	%struct.FogMode = type { %struct.IColor4, float, float, float, float, float, i16, i16, i16, i8, i8 }
	%struct.HintMode = type { i16, i16, i16, i16, i16, i16, i16, i16, i16, i16 }
	%struct.Histogram = type { %struct.ProgramLimits*, i32, i16, i8, i8 }
	%struct.ImagingColorScale = type { %struct.TCoord2, %struct.TCoord2, %struct.TCoord2, %struct.TCoord2 }
	%struct.ImagingSubset = type { %struct.Convolution, %struct.Convolution, %struct.Convolution, %struct.ColorMatrix, %struct.Minmax, %struct.Histogram, %struct.ImagingColorScale, %struct.ImagingColorScale, %struct.ImagingColorScale, %struct.ImagingColorScale, i32, [0 x i32] }
	%struct.Light = type { %struct.IColor4, %struct.IColor4, %struct.IColor4, %struct.IColor4, %struct.PointLineLimits, float, float, float, float, float, %struct.PointLineLimits, float, %struct.PointLineLimits, float, %struct.PointLineLimits, float, float, float, float, float }
	%struct.LightModel = type { %struct.IColor4, [8 x %struct.Light], [2 x %struct.Material], i32, i16, i16, i16, i8, i8, i8, i8, i8, i8 }
	%struct.LightProduct = type { %struct.IColor4, %struct.IColor4, %struct.IColor4 }
	%struct.LineMode = type { float, i32, i16, i16, i8, i8, i8, i8 }
	%struct.LogicOp = type { i16, i8, i8 }
	%struct.MaskMode = type { i32, [3 x i32], i8, i8, i8, i8, i8, i8, i8, i8 }
	%struct.Material = type { %struct.IColor4, %struct.IColor4, %struct.IColor4, %struct.IColor4, float, float, float, float, [8 x %struct.LightProduct], %struct.IColor4, [8 x i32] }
	%struct.Minmax = type { %struct.MinmaxTable*, i16, i8, i8, [0 x i32] }
	%struct.MinmaxTable = type { %struct.IColor4, %struct.IColor4 }
	%struct.Mipmaplevel = type { [4 x i32], [4 x i32], [4 x float], [4 x i32], i32, i32, float*, i8*, i16, i16, i16, i16, [2 x float] }
	%struct.Multisample = type { float, i8, i8, i8, i8, i8, i8, i8, i8 }
	%struct.PipelineProgramState = type { i8, i8, i8, i8, [0 x i32], %struct.IColor4* }
	%struct.PixelMap = type { i32*, float*, float*, float*, float*, float*, float*, float*, float*, i32*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.PixelMode = type { float, float, %struct.PixelStore, %struct.PixelTransfer, %struct.PixelMap, %struct.ImagingSubset, i32, i32 }
	%struct.PixelPack = type { i32, i32, i32, i32, i32, i32, i32, i32, i8, i8, i8, i8 }
	%struct.PixelStore = type { %struct.PixelPack, %struct.PixelPack }
	%struct.PixelTransfer = type { float, float, float, float, float, float, float, float, float, float, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float }
	%struct.PluginBufferData = type { i32 }
	%struct.PointLineLimits = type { float, float, float }
	%struct.PointMode = type { float, float, float, float, %struct.PointLineLimits, float, i8, i8, i8, i8, i16, i16, i32, i16, i16 }
	%struct.PolygonMode = type { [128 x i8], float, float, i16, i16, i16, i16, i8, i8, i8, i8, i8, i8, i8, i8 }
	%struct.ProgramLimits = type { i32, i32, i32, i32 }
	%struct.RegisterCombiners = type { i8, i8, i8, i8, i32, [2 x %struct.IColor4], [8 x %struct.RegisterCombinersPerStageState], %struct.RegisterCombinersFinalStageState }
	%struct.RegisterCombinersFinalStageState = type { i8, i8, i8, i8, [7 x %struct.RegisterCombinersPerVariableState] }
	%struct.RegisterCombinersPerPortionState = type { [4 x %struct.RegisterCombinersPerVariableState], i8, i8, i8, i8, i16, i16, i16, i16, i16, i16 }
	%struct.RegisterCombinersPerStageState = type { [2 x %struct.RegisterCombinersPerPortionState], [2 x %struct.IColor4] }
	%struct.RegisterCombinersPerVariableState = type { i16, i16, i16, i16 }
	%struct.SWRSurfaceRec = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8*, i8*, i8*, [4 x i8*], i32 }
	%struct.ScissorTest = type { %struct.ProgramLimits, i8, i8, i8, i8 }
	%struct.State = type <{ i16, i16, i16, i16, i32, i32, [256 x %struct.IColor4], [128 x %struct.IColor4], %struct.Viewport, %struct.Transform, %struct.LightModel, %struct.ActiveTextureTargets, %struct.AlphaTest, %struct.BlendMode, %struct.ClearColor, %struct.ColorBuffer, %struct.DepthTest, %struct.ArrayRange, %struct.FogMode, %struct.HintMode, %struct.LineMode, %struct.LogicOp, %struct.MaskMode, %struct.PixelMode, %struct.PointMode, %struct.PolygonMode, %struct.ScissorTest, i32, %struct.StencilTest, [8 x %struct.TextureMode], [16 x %struct.TextureImageMode], %struct.ArrayRange, [8 x %struct.TextureCoordGen], %struct.ClipPlane, %struct.Multisample, %struct.RegisterCombiners, %struct.ArrayRange, %struct.ArrayRange, [3 x %struct.PipelineProgramState], %struct.ArrayRange, %struct.TransformFeedback, i32*, %struct.FixedFunction, [3 x i32], [3 x i32] }>
	%struct.StencilTest = type { [3 x { i32, i32, i16, i16, i16, i16 }], i32, [4 x i8] }
	%struct.TextureCoordGen = type { { i16, i16, %struct.IColor4, %struct.IColor4 }, { i16, i16, %struct.IColor4, %struct.IColor4 }, { i16, i16, %struct.IColor4, %struct.IColor4 }, { i16, i16, %struct.IColor4, %struct.IColor4 }, i8, i8, i8, i8 }
	%struct.TextureGeomState = type { i16, i16, i16, i16, i16, i8, i8, i8, i8, i16, i16, i16, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, [6 x i16], [6 x i16] }
	%struct.TextureImageMode = type { float }
	%struct.TextureLevel = type { i32, i32, i16, i16, i16, i8, i8, i16, i16, i16, i16, i8* }
	%struct.TextureMode = type { %struct.IColor4, i32, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, float, float, i16, i16, i16, i16, i16, i16, [4 x i16], i8, i8, i8, i8, [3 x float], [4 x float], float, float }
	%struct.TextureParamState = type { i16, i16, i16, i16, i16, i16, %struct.IColor4, float, float, float, float, i16, i16, i16, i16, float, i16, i8, i8, i32, i8* }
	%struct.TextureRec = type { [4 x float], %struct.TextureState*, %struct.Mipmaplevel*, %struct.Mipmaplevel*, float, float, float, float, i8, i8, i8, i8, i16, i16, i16, i16, i32, float, [2 x %struct.PPStreamToken] }
	%struct.TextureState = type { i16, i8, i8, i16, i16, float, i32, %struct.SWRSurfaceRec*, %struct.TextureParamState, %struct.TextureGeomState, [0 x i32], i8*, i32, %struct.TextureLevel, [1 x [15 x %struct.TextureLevel]] }
	%struct.Transform = type <{ [24 x [16 x float]], [24 x [16 x float]], [16 x float], float, float, float, float, float, i8, i8, i8, i8, i32, i32, i32, i16, i16, i8, i8, i8, i8, i32 }>
	%struct.TransformFeedback = type { i8, i8, i8, i8, [0 x i32], [16 x i32], [16 x i32] }
	%struct.Viewport = type { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, double, double, i32, i32, i32, i32, float, float, float, float }
	%struct.IColor4 = type { float, float, float, float }
	%struct.TCoord2 = type { float, float }
	%struct.VMGPStack = type { [6 x <4 x float>*], <4 x float>*, i32, i32, <4 x float>*, <4 x float>**, i32, i32, i32, i32, i32, i32 }
	%struct.VMTextures = type { [16 x %struct.TextureRec*] }
	%struct.PPStreamToken = type { { i16, i16, i32 } }
	%struct._VMConstants = type { <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, float, float, float, float, float, float, float, float, float, float, float, float, [256 x float], [528 x i8], { void (i8*, i8*, i32, i8*)*, float (float)*, float (float)*, float (float)*, i32 (float)* } }

define i32 @foo(%struct.State* %dst, <4 x float>* %prgrm, <4 x float>** %buffs, %struct._VMConstants* %cnstn, %struct.PPStreamToken* %pstrm, %struct.PluginBufferData* %gpctx, %struct.VMTextures* %txtrs, %struct.VMGPStack* %gpstk) nounwind {
bb266.i:
	getelementptr <4 x float>* null, i32 11		; <<4 x float>*>:0 [#uses=1]
	load <4 x float>* %0, align 16		; <<4 x float>>:1 [#uses=1]
	shufflevector <4 x float> %1, <4 x float> undef, <4 x i32> < i32 0, i32 1, i32 1, i32 1 >		; <<4 x float>>:2 [#uses=1]
	shufflevector <4 x float> %2, <4 x float> undef, <4 x i32> < i32 0, i32 4, i32 1, i32 5 >		; <<4 x float>>:3 [#uses=1]
	shufflevector <4 x float> undef, <4 x float> undef, <4 x i32> < i32 0, i32 4, i32 1, i32 5 >		; <<4 x float>>:4 [#uses=1]
	shufflevector <4 x float> %4, <4 x float> %3, <4 x i32> < i32 6, i32 7, i32 2, i32 3 >		; <<4 x float>>:5 [#uses=1]
	mul <4 x float> %5, zeroinitializer		; <<4 x float>>:6 [#uses=2]
	mul <4 x float> %6, %6		; <<4 x float>>:7 [#uses=1]
	add <4 x float> zeroinitializer, %7		; <<4 x float>>:8 [#uses=1]
	call <4 x float> @llvm.x86.sse.max.ps( <4 x float> zeroinitializer, <4 x float> %8 ) nounwind readnone		; <<4 x float>>:9 [#uses=1]
	%phitmp40 = bitcast <4 x float> %9 to <4 x i32>		; <<4 x i32>> [#uses=1]
	%tmp4109.i = and <4 x i32> %phitmp40, < i32 8388607, i32 8388607, i32 8388607, i32 8388607 >		; <<4 x i32>> [#uses=1]
	%tmp4116.i = or <4 x i32> %tmp4109.i, < i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216 >		; <<4 x i32>> [#uses=1]
	%tmp4117.i = bitcast <4 x i32> %tmp4116.i to <4 x float>		; <<4 x float>> [#uses=1]
	add <4 x float> %tmp4117.i, zeroinitializer		; <<4 x float>>:10 [#uses=1]
	mul <4 x float> %10, < float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01 >		; <<4 x float>>:11 [#uses=1]
	call <4 x float> @llvm.x86.sse.max.ps( <4 x float> %11, <4 x float> zeroinitializer ) nounwind readnone		; <<4 x float>>:12 [#uses=1]
	call <4 x float> @llvm.x86.sse.min.ps( <4 x float> %12, <4 x float> zeroinitializer ) nounwind readnone		; <<4 x float>>:13 [#uses=1]
	%tmp4170.i = call <4 x float> @llvm.x86.sse.cmp.ps( <4 x float> %13, <4 x float> zeroinitializer, i8 2 ) nounwind		; <<4 x float>> [#uses=1]
	bitcast <4 x float> %tmp4170.i to <16 x i8>		; <<16 x i8>>:14 [#uses=1]
	call i32 @llvm.x86.sse2.pmovmskb.128( <16 x i8> %14 ) nounwind readnone		; <i32>:15 [#uses=1]
	icmp eq i32 %15, 0		; <i1>:16 [#uses=1]
	br i1 %16, label %bb5574.i, label %bb4521.i

bb4521.i:		; preds = %bb266.i
	unreachable

bb5574.i:		; preds = %bb266.i
	unreachable
}

declare <4 x float> @llvm.x86.sse.cmp.ps(<4 x float>, <4 x float>, i8) nounwind readnone

declare i32 @llvm.x86.sse2.pmovmskb.128(<16 x i8>) nounwind readnone

declare <4 x float> @llvm.x86.sse.max.ps(<4 x float>, <4 x float>) nounwind readnone

declare <4 x float> @llvm.x86.sse.min.ps(<4 x float>, <4 x float>) nounwind readnone
