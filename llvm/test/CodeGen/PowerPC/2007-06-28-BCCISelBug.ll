; RUN: llvm-as < %s | llc -march=ppc32 -mattr=+altivec

	%struct.XATest = type { float, i16, i8, i8 }
	%struct.XArrayRange = type { i8, i8, i8, i8 }
	%struct.XBlendMode = type { i16, i16, i16, i16, %struct.GIC4, i16, i16, i8, i8, i8, i8 }
	%struct.XClearC = type { double, %struct.GIC4, %struct.GIC4, float, i32 }
	%struct.XClipPlane = type { i32, [6 x %struct.GIC4] }
	%struct.XCBuffer = type { i16, i16, [8 x i16] }
	%struct.XCMatrix = type { [16 x float]*, %struct.XICSS }
	%struct.XConvolution = type { %struct.GIC4, %struct.XICSS, i16, i16, float*, i32, i32 }
	%struct.XDepthTest = type { i16, i16, i8, i8, i8, i8, double, double }
	%struct.XFixedFunctionProgram = type { %struct.PPSToken* }
	%struct.XFogMode = type { %struct.GIC4, float, float, float, float, float, i16, i16, i16, i8, i8 }
	%struct.XFramebufferAttachment = type { i32, i32, i32, i32 }
	%struct.XHintMode = type { i16, i16, i16, i16, i16, i16, i16, i16, i16, i16 }
	%struct.XHistogram = type { %struct.XFramebufferAttachment*, i32, i16, i8, i8 }
	%struct.XICSS = type { %struct.GTCoord2, %struct.GTCoord2, %struct.GTCoord2, %struct.GTCoord2 }
	%struct.XISubset = type { %struct.XConvolution, %struct.XConvolution, %struct.XConvolution, %struct.XCMatrix, %struct.XMinmax, %struct.XHistogram, %struct.XICSS, %struct.XICSS, %struct.XICSS, %struct.XICSS, i32 }
	%struct.XLight = type { %struct.GIC4, %struct.GIC4, %struct.GIC4, %struct.GIC4, %struct.XPointLineLimits, float, float, float, float, float, %struct.XPointLineLimits, float, float, float, float, float }
	%struct.XLightModel = type { %struct.GIC4, [8 x %struct.XLight], [2 x %struct.XMaterial], i32, i16, i16, i16, i8, i8, i8, i8, i8, i8 }
	%struct.XLightProduct = type { %struct.GIC4, %struct.GIC4, %struct.GIC4 }
	%struct.XLineMode = type { float, i32, i16, i16, i8, i8, i8, i8 }
	%struct.XLogicOp = type { i16, i8, i8 }
	%struct.XMaskMode = type { i32, [3 x i32], i8, i8, i8, i8, i8, i8, i8, i8 }
	%struct.XMaterial = type { %struct.GIC4, %struct.GIC4, %struct.GIC4, %struct.GIC4, float, float, float, float, [8 x %struct.XLightProduct], %struct.GIC4, [6 x i32], [2 x i32] }
	%struct.XMinmax = type { %struct.XMinmaxTable*, i16, i8, i8 }
	%struct.XMinmaxTable = type { %struct.GIC4, %struct.GIC4 }
	%struct.XMipmaplevel = type { [4 x i32], [4 x i32], [4 x float], [4 x i32], i32, i32, float*, i8*, i16, i16, i16, i16, [2 x float] }
	%struct.XMultisample = type { float, i8, i8, i8, i8, i8, i8, i8, i8 }
	%struct.XPipelineProgramState = type { i8, i8, i8, i8, %struct.GIC4* }
	%struct.XPMap = type { i32*, float*, float*, float*, float*, float*, float*, float*, float*, i32*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.XPMode = type { float, float, %struct.XPStore, %struct.XPTransfer, %struct.XPMap, %struct.XISubset, i32, i32 }
	%struct.XPPack = type { i32, i32, i32, i32, i32, i32, i32, i32, i8, i8, i8, i8 }
	%struct.XPStore = type { %struct.XPPack, %struct.XPPack }
	%struct.XPTransfer = type { float, float, float, float, float, float, float, float, float, float, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float }
	%struct.XPointLineLimits = type { float, float, float }
	%struct.XPointMode = type { float, float, float, float, %struct.XPointLineLimits, float, i8, i8, i8, i8, i16, i16, i32, i16, i16 }
	%struct.XPGMode = type { [128 x i8], float, float, i16, i16, i16, i16, i8, i8, i8, i8, i8, i8, i8, i8 }
	%struct.XRegisterCCs = type { i8, i8, i8, i8, i32, [2 x %struct.GIC4], [8 x %struct.XRegisterCCsPerStageState], %struct.XRegisterCCsFinalStageState }
	%struct.XRegisterCCsFinalStageState = type { i8, i8, i8, i8, [7 x %struct.XRegisterCCsPerVariableState] }
	%struct.XRegisterCCsPerPortionState = type { [4 x %struct.XRegisterCCsPerVariableState], i8, i8, i8, i8, i16, i16, i16, i16, i16, i16 }
	%struct.XRegisterCCsPerStageState = type { [2 x %struct.XRegisterCCsPerPortionState], [2 x %struct.GIC4] }
	%struct.XRegisterCCsPerVariableState = type { i16, i16, i16, i16 }
	%struct.XScissorTest = type { %struct.XFramebufferAttachment, i8, i8, i8, i8 }
	%struct.XState = type { i16, i16, i16, i16, i32, i32, [256 x %struct.GIC4], [128 x %struct.GIC4], %struct.XViewport, %struct.XXF, %struct.XLightModel, %struct.XATest, %struct.XBlendMode, %struct.XClearC, %struct.XCBuffer, %struct.XDepthTest, %struct.XArrayRange, %struct.XFogMode, %struct.XHintMode, %struct.XLineMode, %struct.XLogicOp, %struct.XMaskMode, %struct.XPMode, %struct.XPointMode, %struct.XPGMode, %struct.XScissorTest, i32, %struct.XStencilTest, [16 x %struct.XTMode], %struct.XArrayRange, [8 x %struct.XTCoordGen], %struct.XClipPlane, %struct.XMultisample, %struct.XRegisterCCs, %struct.XArrayRange, %struct.XArrayRange, [3 x %struct.XPipelineProgramState], %struct.XXFFeedback, i32*, %struct.XFixedFunctionProgram, [3 x i32] }
	%struct.XStencilTest = type { [3 x { i32, i32, i16, i16, i16, i16 }], i32, [4 x i8] }
	%struct.XTCoordGen = type { { i16, i16, %struct.GIC4, %struct.GIC4 }, { i16, i16, %struct.GIC4, %struct.GIC4 }, { i16, i16, %struct.GIC4, %struct.GIC4 }, { i16, i16, %struct.GIC4, %struct.GIC4 }, i8, i8, i8, i8 }
	%struct.XTGeomState = type { i16, i16, i16, i16, i16, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, [6 x i16], [6 x i16] }
	%struct.XTLevel = type { i32, i32, i16, i16, i16, i8, i8, i16, i16, i16, i16, i8* }
	%struct.XTMode = type { %struct.GIC4, i32, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, float, float, float, i16, i16, i16, i16, i16, i16, [4 x i16], i8, i8, i8, i8, [3 x float], [4 x float], float, float }
	%struct.XTParamState = type { i16, i16, i16, i16, i16, i16, %struct.GIC4, float, float, float, float, i16, i16, i16, i16, float, i16, i8, i8, i32, i8* }
	%struct.XTRec = type { %struct.XTState*, float, float, float, float, %struct.XMipmaplevel*, %struct.XMipmaplevel*, i32, i32, i32, i32, i32, i32, i32, [2 x %struct.PPSToken] }
	%struct.XTState = type { i16, i8, i8, i16, i16, float, i32, %struct.GISWRSurface*, %struct.XTParamState, %struct.XTGeomState, %struct.XTLevel, [6 x [15 x %struct.XTLevel]] }
	%struct.XXF = type { [24 x [16 x float]], [24 x [16 x float]], [16 x float], float, float, float, float, float, i8, i8, i8, i8, i32, i32, i32, i16, i16, i8, i8, i8, i8, i32 }
	%struct.XXFFeedback = type { i8, i8, i8, i8, [16 x i32], [16 x i32] }
	%struct.XViewport = type { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, double, double, i32, i32, i32, i32, float, float, float, float }
	%struct.GIC4 = type { float, float, float, float }
	%struct.GISWRSurface = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8*, i8*, i8*, [4 x i8*], i32 }
	%struct.GTCoord2 = type { float, float }
	%struct.GVMFPContext = type { float, i32, i32, i32, float, [3 x float] }
	%struct.GVMFPStack = type { [8 x i8*], i8*, i8*, i32, i32, { <4 x float> }, { <4 x float> }, <4 x i32> }
	%struct.GVMFGAttrib = type { <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, [8 x <4 x float>] }
	%struct.GVMTs = type { [16 x %struct.XTRec*] }
	%struct.PPSToken = type { { i16, i16, i32 } }
	%struct._GVMConstants = type { <4 x i32>, <4 x i32>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float>, float, float, float, float, float, float, float, float, float, float, float, float, [256 x float], [528 x i8] }

declare <4 x i32> @llvm.ppc.altivec.lvewx(i8*)

declare i32 @llvm.ppc.altivec.vcmpequw.p(i32, <4 x i32>, <4 x i32>)

define void @test(%struct.XState* %gldst, <4 x float>* %prgrm, <4 x float>** %buffs, %struct._GVMConstants* %cnstn, %struct.PPSToken* %pstrm, %struct.GVMFPContext* %vmctx, %struct.GVMTs* %txtrs, %struct.GVMFPStack* %fpstk, %struct.GVMFGAttrib* %start, %struct.GVMFGAttrib* %deriv, i32 %fragx, i32 %fragy) {
bb58.i:
	%tmp3405.i = getelementptr %struct.XTRec* null, i32 0, i32 1		; <float*> [#uses=1]
	%tmp34053406.i = bitcast float* %tmp3405.i to i8*		; <i8*> [#uses=1]
	%tmp3407.i = call <4 x i32> @llvm.ppc.altivec.lvewx( i8* %tmp34053406.i )		; <<4 x i32>> [#uses=0]
	%tmp4146.i = call i32 @llvm.ppc.altivec.vcmpequw.p( i32 3, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer )		; <i32> [#uses=1]
	%tmp4147.i = icmp eq i32 %tmp4146.i, 0		; <i1> [#uses=1]
	br i1 %tmp4147.i, label %bb8799.i, label %bb4150.i

bb4150.i:		; preds = %bb58.i
	br label %bb8799.i

bb8799.i:		; preds = %bb4150.i, %bb58.i
	ret void
}
