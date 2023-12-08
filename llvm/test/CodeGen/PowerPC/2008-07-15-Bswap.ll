; RUN: llc -verify-machineinstrs < %s | FileCheck %s

target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f128:64:128"
target triple = "powerpc-unknown-linux-gnu"
	%struct.BiPartSrcDescriptor = type <{ %"struct.BiPartSrcDescriptor::$_105" }>
	%"struct.BiPartSrcDescriptor::$_105" = type { %struct.BiPartSrcDescriptor_NO_VECTOR_ALIGNMENT_size_is_16 }
	%struct.BiPartSrcDescriptor_NO_VECTOR_ALIGNMENT_size_is_16 = type { [2 x %struct.MotionVectors], [2 x i8], %struct.Map4x4ToPartIdx, [2 x i8], i8, i8 }
	%struct.Condv = type opaque
	%struct.DHBFLayerId = type { i8 }
	%struct.DecodeComplexityInfo = type { i32, i32, i32, i32, %"struct.DecodeComplexityInfo::IntraStats", %"struct.DecodeComplexityInfo::InterStats" }
	%"struct.DecodeComplexityInfo::InterStats" = type { i32, i32, i32, i32, [5 x i32], [3 x i32], [4 x [4 x i32]], [4 x i32], i32, %struct.MotionVectors, %struct.MotionVectors }
	%"struct.DecodeComplexityInfo::IntraStats" = type { i32, i32, i32, [5 x i32], [3 x i32], [4 x i32], [3 x i32] }
	%struct.DecodeComplexityOptions = type { i8, i8, i32, double, i8, float, i8, float, i8, i8, i8, i8, i8 }
	%struct.DescriptorAllocator = type { ptr, ptr, ptr, i32, i32, ptr, i32, i32, ptr, i32, i32 }
	%struct.DetailsFromSliceType = type <{ i8 }>
	%struct.FlatnessAnalysis = type { i16, i16, i32, ptr, ptr, [512 x i32], [256 x i32] }
	%struct.Frame = type <{ i8, i8, i8, i8, i8, [3 x i8], i32, i32, ptr, ptr, [8 x i8], %struct.FramePixels, %struct.FrameMotionVectorCache, %struct.FrameIndex, i32, ptr, ptr, ptr, ptr, ptr, %struct.FlatnessAnalysis, %struct.NoiseAnalysis, %struct.VisualActivity, %struct.FrameMotionInfo, %struct.FrameMotionAnalysis, %struct.FrameDataRateParameters, %struct.FrameEncoderTags, %struct.DecodeComplexityInfo, %struct.DecodeComplexityOptions, ptr, [1 x i32] }>
	%struct.FrameDataRateParameters = type { i32, float, i8, i8 }
	%struct.FrameEncoderTags = type { i8, i8, i32, i8, i8, float }
	%struct.FrameIndex = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i8, i8, i8, i32, i32, ptr, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, %struct.DHBFLayerId }
	%struct.FrameMotionAnalysis = type { i32, i32, i32, ptr, ptr, i32, i32, i16, i16, i32, i32, i32, i32, i8, i8 }
	%struct.FrameMotionInfo = type { i32, i32, ptr, i32, i32, i32, i32, i32 }
	%struct.FrameMotionVectorCache = type <{ ptr, i32, i32, i32, %struct.BiPartSrcDescriptor, %struct.BiPartSrcDescriptor, %struct.BiPartSrcDescriptor, [3 x ptr], ptr }>
	%struct.FramePixels = type <{ i8, i8, i8, i8, i8, i8, i8, i8, ptr, ptr, i32, [4 x ptr], [4 x ptr], [2 x [4 x i32]], [2 x [4 x i32]], %struct.PixelData, ptr, ptr, ptr, [16 x i16], [16 x i16], [12 x i8], %"struct.PortableSInt32Array<4>", %"struct.PortableSInt32Array<8>", %struct.ICOffsetArraysY, ptr, ptr, ptr, [3 x i32] }>
	%struct.ICOffsetArraysY = type { [21 x i32], [21 x i32], [4 x [21 x i32]] }
	%struct.InterpolationCache = type opaque
	%struct.LoopFilterInfo = type { ptr, i32, i32, i32, i32, ptr, i32, ptr, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8, i8, i8, ptr, ptr, i8, ptr, ptr, i32 }
	%"struct.LoopFilterInfo::SliceInfoStruct" = type { %"struct.LoopFilterInfo::SliceInfoStruct::LFDisableStats", i8, i8, i8, i8, [17 x ptr], [17 x ptr] }
	%"struct.LoopFilterInfo::SliceInfoStruct::LFDisableStats" = type { i32, i32 }
	%struct.LoopFilterParam = type { i32, ptr, ptr, ptr }
	%struct.Map4x4ToPartIdx = type { i16 }
	%struct.MbAnalysis = type { i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, %struct.RdCost, %struct.RdCost, i32 }
	%struct.MoEstMbMotionInfo = type { i32, i32, i32, i32, [16 x %struct.MoEstPartMotionInfo] }
	%struct.MoEstMotion16x16 = type { [2 x i8], [2 x %struct.MotionVectors], i8, [3 x %struct.MoEstPredCost] }
	%struct.MoEstPartMotionInfo = type { i32, %struct.PartGeom, i32, i32, [2 x %struct.MotionVectors], [2 x i8], i16 }
	%struct.MoEstPredCost = type { i32, i16, i16 }
	%struct.MotionInfoFor16x16_FasterSP = type { [2 x %struct.MotionVectors], [2 x i8], i8, [2 x i32], i32, i32 }
	%struct.MotionVectors = type { %"struct.MotionVectors::$_103" }
	%"struct.MotionVectors::$_103" = type { i32 }
	%struct.Mutex = type opaque
	%struct.NoiseAnalysis = type { i16, i16, i32, ptr, ptr, ptr, [512 x i32] }
	%struct.PartGeom = type { %struct.Map4x4ToPartIdx }
	%struct.PerMacroblockBoundaryStrengths = type { [16 x i8], [16 x i8], [4 x i8], [4 x i8], [2 x i32] }
	%struct.PixelData = type { ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8, i8 }
	%"struct.PortableSInt32Array<4>" = type { [4 x i32] }
	%"struct.PortableSInt32Array<8>" = type { [8 x i32] }
	%struct.RdCost = type { i32, i32, i32, double }
	%struct.ThreadAllocator = type { ptr, ptr, [256 x ptr], i32, i32, i32 }
	%struct.ThreadedBatch = type opaque
	%struct.UVSrcOffsetEtcX_Struct = type <{ i16 }>
	%struct.VisualActivity = type { i16, i16, i32, i32, ptr, ptr, i32, i32, ptr, i32, i32, i32, i32, i32, ptr, i32, [2 x i32], i32, i32, i32, ptr, i16, i16, i16, i16, float, ptr, ptr, i32, i32, i8 }
@_ZL33table_8_14_indexA_to_alpha_scalar = external constant [64 x i8]		; <ptr> [#uses=0]
@_ZL32table_8_14_indexB_to_beta_scalar = external constant [64 x i8]		; <ptr> [#uses=0]
@_ZL34table_8_15_indexA_bS_to_tc0_scalar = external constant [64 x [4 x i8]]		; <ptr> [#uses=0]
@gkDummy = external global i32		; <ptr> [#uses=0]
@gkDetailsFromSliceTypeArray = external constant [10 x %struct.DetailsFromSliceType]		; <ptr> [#uses=0]

declare i32 @_Z20LoopFilter_ConstructP14LoopFilterInfojj(ptr, i32, i32)

declare i32 @_Z25LF_Threading2_assert_doneP14LoopFilterInfo(ptr) nounwind 

declare i32 @_Z54S_CalcIfLargeMVDeltaForBMbBothPredictionsFromSameFramePK19BiPartSrcDescriptorS1_ijj(ptr, ptr, i32, i32, i32) nounwind 

declare void @_Z30LoopFilter_Internal_FilterLumaPhiiiiii(ptr, i32, i32, i32, i32, i32, i32) nounwind 

declare void @_Z33LoopFilter_Internal_FilterChromaVPhiiiiiiiiii(ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) nounwind 

declare void @_Z33LoopFilter_Internal_FilterChromaHPhiiiiii(ptr, i32, i32, i32, i32, i32, i32) nounwind 

declare void @_Z42LoopFilter_Internal_filter_macroblock_lumaPK14LoopFilterInfoPhS2_iiiPK30PerMacroblockBoundaryStrengthsjj(ptr, ptr, ptr, i32, i32, i32, ptr, i32, i32) nounwind 

declare i32 @_Z40LoopFilter_Internal_FilterLumaPlaneMBAFFPK14LoopFilterInfojjj(ptr, i32, i32, i32) nounwind 

declare void @_Z18LoopFilter_DestroyP14LoopFilterInfo(ptr)

declare void @MutexDispose(ptr)

declare void @_ZdaPv(ptr) nounwind 

declare void @jvtDisposePTRVectorAligned(ptr)

declare void @jvtDisposePTR(ptr)

declare void @jvtDisposePTRMemAligned(ptr)

declare void @_Z31LoopFilter_Internal_ResetTablesP14LoopFilterInfo(ptr) nounwind 

; CHECK: _Z60LoopFilter_Internal_CalculateBoundaryStrengths_MbaffFramePicPK14LoopFilterInfoP22FrameMotionVectorCachejj
; CHECK: blr

define i32 @_Z60LoopFilter_Internal_CalculateBoundaryStrengths_MbaffFramePicPK14LoopFilterInfoP22FrameMotionVectorCachejj(ptr %lfiPtr, ptr %frameMotionVectorCachePtr, i32 %mbY_min, i32 %mbY_maxPlus1) nounwind  {
entry:
	icmp ult i32 %mbY_min, %mbY_maxPlus1		; <i1>:0 [#uses=1]
	br i1 %0, label %bb16, label %bb642

bb16:		; preds = %entry
	bitcast ptr null to ptr		; <ptr>:1 [#uses=3]
	getelementptr i32, ptr %1, i32 1		; <ptr>:2 [#uses=0]
	getelementptr i32, ptr %1, i32 2		; <ptr>:3 [#uses=0]
	getelementptr i32, ptr %1, i32 3		; <ptr>:4 [#uses=0]
	bitcast ptr null to ptr		; <ptr>:5 [#uses=3]
	getelementptr i32, ptr %5, i32 1		; <ptr>:6 [#uses=0]
	getelementptr i32, ptr %5, i32 2		; <ptr>:7 [#uses=0]
	getelementptr i32, ptr %5, i32 3		; <ptr>:8 [#uses=0]
	icmp eq i32 0, 0		; <i1>:9 [#uses=0]
	lshr i32 0, 30		; <i32>:10 [#uses=0]
	and i32 0, 268435455		; <i32>:11 [#uses=0]
	lshr i32 0, 28		; <i32>:12 [#uses=1]
	and i32 %12, 3		; <i32>:13 [#uses=0]
	and i32 0, 1		; <i32>:14 [#uses=1]
	icmp eq i32 %14, 0		; <i1>:15 [#uses=0]
	zext i8 0 to i32		; <i32>:16 [#uses=1]
	%.not656 = icmp ne i32 0, 0		; <i1> [#uses=1]
	icmp eq i8 0, 0		; <i1>:17 [#uses=0]
	trunc i32 0 to i8		; <i8>:18 [#uses=2]
	add i32 0, 1		; <i32>:19 [#uses=1]
	%.not658 = icmp ne i32 0, 0		; <i1> [#uses=1]
	and i32 0, 268369920		; <i32>:20 [#uses=1]
	icmp eq i32 %20, 268369920		; <i1>:21 [#uses=2]
	getelementptr %struct.PerMacroblockBoundaryStrengths, ptr null, i32 0, i32 2		; <ptr>:22 [#uses=1]
	getelementptr %struct.PerMacroblockBoundaryStrengths, ptr null, i32 0, i32 2, i32 0		; <ptr>:23 [#uses=0]
	and i32 0, -2		; <i32>:24 [#uses=1]
	add i32 %24, -1		; <i32>:25 [#uses=0]
	bitcast ptr %22 to ptr		; <ptr>:26 [#uses=3]
	getelementptr i32, ptr %26, i32 1		; <ptr>:27 [#uses=0]
	getelementptr i32, ptr %26, i32 2		; <ptr>:28 [#uses=0]
	getelementptr i32, ptr %26, i32 3		; <ptr>:29 [#uses=0]
	br label %bb144

bb144:		; preds = %bb395, %bb16
	%idxEachField11.0773 = phi i32 [ 0, %bb16 ], [ %162, %bb395 ]		; <i32> [#uses=3]
	%mbYLeft.2776 = phi i32 [ 0, %bb16 ], [ %mbYLeft.2776, %bb395 ]		; <i32> [#uses=3]
	%mbXYLeft.2775 = phi i32 [ 0, %bb16 ], [ %mbXYLeft.2775, %bb395 ]		; <i32> [#uses=1]
	%mixedModeLeftEdgeOfMbFlag.2774 = phi i32 [ 0, %bb16 ], [ 0, %bb395 ]		; <i32> [#uses=0]
	%mbIndexLeft.2772 = phi i32 [ 0, %bb16 ], [ %mbIndexLeft.2772, %bb395 ]		; <i32> [#uses=2]
	%boundaryStrengthsV.1771 = phi ptr [ null, %bb16 ], [ %158, %bb395 ]		; <ptr> [#uses=2]
	%numEdgesToTest.1770 = phi i32 [ 4, %bb16 ], [ %numEdgesToTest.2, %bb395 ]		; <i32> [#uses=1]
	icmp eq i32 %idxEachField11.0773, 0		; <i1>:30 [#uses=0]
	getelementptr ptr, ptr null, i32 %mbIndexLeft.2772		; <ptr>:31 [#uses=1]
	load ptr, ptr %31, align 4		; <ptr>:32 [#uses=0]
	%fMacroblockHasNonZeroBS.4 = select i1 %21, i32 1, i32 0		; <i32> [#uses=1]
	%numEdgesToTest.2 = select i1 %21, i32 1, i32 %numEdgesToTest.1770		; <i32> [#uses=2]
	store i8 32, ptr %boundaryStrengthsV.1771, align 1
	br label %labelContinueEdgesLoopV

bb200:		; preds = %labelContinueEdgesLoopV
	lshr i32 %159, 28		; <i32>:33 [#uses=2]
	and i32 %160, %16		; <i32>:34 [#uses=1]
	icmp eq i32 %34, 0		; <i1>:35 [#uses=0]
	icmp eq i32 %160, 0		; <i1>:36 [#uses=3]
	zext i1 %36 to i32		; <i32>:37 [#uses=1]
	or i32 %37, -1		; <i32>:38 [#uses=1]
	or i32 %38, %33		; <i32>:39 [#uses=1]
	icmp eq i32 %39, 0		; <i1>:40 [#uses=1]
	br i1 %40, label %bb205, label %bb206

bb205:		; preds = %bb200
	store i8 32, ptr %158, align 1
	br label %labelContinueEdgesLoopV

bb206:		; preds = %bb200
	icmp eq i32 %33, 15		; <i1>:41 [#uses=1]
	br i1 %41, label %labelContinueEdgesLoopV, label %bb210.preheader

bb210.preheader:		; preds = %bb206
	add i32 %160, 0		; <i32>:42 [#uses=2]
	%bothcond657 = and i1 %36, %.not656		; <i1> [#uses=0]
	shl i32 %idxEachField11.0773, 1		; <i32>:43 [#uses=1]
	add i32 %43, 0		; <i32>:44 [#uses=0]
	shl i32 %mbYLeft.2776, 2		; <i32>:45 [#uses=0]
	add i32 %42, -1		; <i32>:46 [#uses=1]
	icmp eq i32 0, 0		; <i1>:47 [#uses=1]
	%brmerge689.not = and i1 %47, false		; <i1> [#uses=0]
	%bothcond659 = and i1 %36, %.not658		; <i1> [#uses=0]
	shl i32 %mbYLeft.2776, 1		; <i32>:48 [#uses=1]
	or i32 %48, 0		; <i32>:49 [#uses=1]
	shl i32 %49, 1		; <i32>:50 [#uses=0]
	add i32 0, 0		; <i32>:51 [#uses=2]
	mul i32 %51, 0		; <i32>:52 [#uses=1]
	add i32 %52, %42		; <i32>:53 [#uses=1]
	mul i32 %51, 0		; <i32>:54 [#uses=1]
	add i32 %46, %54		; <i32>:55 [#uses=1]
	getelementptr ptr, ptr null, i32 %53		; <ptr>:56 [#uses=1]
	load ptr, ptr %56, align 4		; <ptr>:57 [#uses=7]
	getelementptr ptr, ptr null, i32 %55		; <ptr>:58 [#uses=1]
	load ptr, ptr %58, align 4		; <ptr>:59 [#uses=5]
	icmp slt i32 %159, 0		; <i1>:60 [#uses=0]
	icmp eq ptr %57, %59		; <i1>:61 [#uses=0]
	bitcast ptr %57 to ptr		; <ptr>:62 [#uses=5]
	load i16, ptr %62, align 2		; <i16>:63 [#uses=2]
	getelementptr i16, ptr %62, i32 1		; <ptr>:64 [#uses=1]
	load i16, ptr %64, align 2		; <i16>:65 [#uses=2]
	getelementptr i16, ptr %62, i32 2		; <ptr>:66 [#uses=1]
	load i16, ptr %66, align 2		; <i16>:67 [#uses=2]
	getelementptr i16, ptr %62, i32 3		; <ptr>:68 [#uses=1]
	load i16, ptr %68, align 2		; <i16>:69 [#uses=2]
	getelementptr i16, ptr %62, i32 6		; <ptr>:70 [#uses=1]
	load i16, ptr %70, align 2		; <i16>:71 [#uses=2]
	bitcast ptr %59 to ptr		; <ptr>:72 [#uses=5]
	load i16, ptr %72, align 2		; <i16>:73 [#uses=2]
	getelementptr i16, ptr %72, i32 1		; <ptr>:74 [#uses=1]
	load i16, ptr %74, align 2		; <i16>:75 [#uses=2]
	getelementptr i16, ptr %72, i32 2		; <ptr>:76 [#uses=1]
	load i16, ptr %76, align 2		; <i16>:77 [#uses=2]
	getelementptr i16, ptr %72, i32 3		; <ptr>:78 [#uses=1]
	load i16, ptr %78, align 2		; <i16>:79 [#uses=2]
	getelementptr i16, ptr %72, i32 6		; <ptr>:80 [#uses=1]
	load i16, ptr %80, align 2		; <i16>:81 [#uses=2]
	sub i16 %63, %73		; <i16>:82 [#uses=3]
	sub i16 %65, %75		; <i16>:83 [#uses=3]
	sub i16 %67, %77		; <i16>:84 [#uses=3]
	sub i16 %69, %79		; <i16>:85 [#uses=3]
	sub i16 %71, %81		; <i16>:86 [#uses=3]
	sub i16 0, %82		; <i16>:87 [#uses=1]
	icmp slt i16 %82, 0		; <i1>:88 [#uses=1]
	%. = select i1 %88, i16 %87, i16 %82		; <i16> [#uses=1]
	sub i16 0, %83		; <i16>:89 [#uses=1]
	icmp slt i16 %83, 0		; <i1>:90 [#uses=1]
	%.660 = select i1 %90, i16 %89, i16 %83		; <i16> [#uses=1]
	sub i16 0, %84		; <i16>:91 [#uses=1]
	icmp slt i16 %84, 0		; <i1>:92 [#uses=1]
	%.661 = select i1 %92, i16 %91, i16 %84		; <i16> [#uses=1]
	sub i16 0, %85		; <i16>:93 [#uses=1]
	icmp slt i16 %85, 0		; <i1>:94 [#uses=1]
	%.662 = select i1 %94, i16 %93, i16 %85		; <i16> [#uses=1]
	sub i16 0, %86		; <i16>:95 [#uses=1]
	icmp slt i16 %86, 0		; <i1>:96 [#uses=1]
	%.663 = select i1 %96, i16 %95, i16 %86		; <i16> [#uses=1]
	getelementptr %struct.BiPartSrcDescriptor, ptr %57, i32 0, i32 0, i32 0, i32 1, i32 0		; <ptr>:97 [#uses=1]
	load i8, ptr %97, align 1		; <i8>:98 [#uses=1]
	zext i8 %98 to i32		; <i32>:99 [#uses=1]
	getelementptr %struct.BiPartSrcDescriptor, ptr %57, i32 0, i32 0, i32 0, i32 1, i32 1		; <ptr>:100 [#uses=1]
	load i8, ptr %100, align 1		; <i8>:101 [#uses=1]
	zext i8 %101 to i32		; <i32>:102 [#uses=1]
	getelementptr %struct.BiPartSrcDescriptor, ptr %57, i32 0, i32 0, i32 0, i32 3, i32 0		; <ptr>:103 [#uses=1]
	load i8, ptr %103, align 1		; <i8>:104 [#uses=2]
	zext i8 %104 to i32		; <i32>:105 [#uses=1]
	getelementptr %struct.BiPartSrcDescriptor, ptr %59, i32 0, i32 0, i32 0, i32 3, i32 0		; <ptr>:106 [#uses=1]
	load i8, ptr %106, align 1		; <i8>:107 [#uses=2]
	zext i8 %107 to i32		; <i32>:108 [#uses=1]
	getelementptr %struct.BiPartSrcDescriptor, ptr %57, i32 0, i32 0, i32 0, i32 3, i32 1		; <ptr>:109 [#uses=1]
	load i8, ptr %109, align 1		; <i8>:110 [#uses=1]
	zext i8 %110 to i32		; <i32>:111 [#uses=1]
	getelementptr %struct.BiPartSrcDescriptor, ptr %59, i32 0, i32 0, i32 0, i32 3, i32 1		; <ptr>:112 [#uses=1]
	load i8, ptr %112, align 1		; <i8>:113 [#uses=1]
	zext i8 %113 to i32		; <i32>:114 [#uses=1]
	lshr i32 %99, 4		; <i32>:115 [#uses=1]
	and i32 %115, 2		; <i32>:116 [#uses=1]
	lshr i32 %102, 5		; <i32>:117 [#uses=1]
	or i32 %116, %117		; <i32>:118 [#uses=3]
	icmp eq i32 %118, 0		; <i1>:119 [#uses=0]
	icmp eq i32 %118, 1		; <i1>:120 [#uses=1]
	br i1 %120, label %bb297, label %bb298

bb297:		; preds = %bb210.preheader
	br label %bb298

bb298:		; preds = %bb297, %bb210.preheader
	%vu8Mask_0.1 = phi i8 [ -1, %bb297 ], [ 0, %bb210.preheader ]		; <i8> [#uses=1]
	%vu8Mask_1.1 = phi i8 [ -1, %bb297 ], [ 0, %bb210.preheader ]		; <i8> [#uses=1]
	%vu8Mask_2.1 = phi i8 [ -1, %bb297 ], [ 0, %bb210.preheader ]		; <i8> [#uses=0]
	%vu8Mask_3.1 = phi i8 [ -1, %bb297 ], [ 0, %bb210.preheader ]		; <i8> [#uses=1]
	%vu8Mask_4.1 = phi i8 [ 0, %bb297 ], [ 0, %bb210.preheader ]		; <i8> [#uses=0]
	%vu8Mask_5.1 = phi i8 [ 0, %bb297 ], [ 0, %bb210.preheader ]		; <i8> [#uses=1]
	%vu8Mask_6.1 = phi i8 [ 0, %bb297 ], [ 0, %bb210.preheader ]		; <i8> [#uses=0]
	%vu8Mask_7.1 = phi i8 [ 0, %bb297 ], [ 0, %bb210.preheader ]		; <i8> [#uses=1]
	%vu8Mask_12.1 = phi i8 [ -1, %bb297 ], [ 0, %bb210.preheader ]		; <i8> [#uses=0]
	%vu8Mask_13.1 = phi i8 [ -1, %bb297 ], [ 0, %bb210.preheader ]		; <i8> [#uses=0]
	icmp eq i32 %118, 2		; <i1>:121 [#uses=0]
	and i8 %vu8Mask_1.1, 3		; <i8>:122 [#uses=0]
	and i8 %vu8Mask_5.1, 3		; <i8>:123 [#uses=0]
	and i8 %vu8Mask_3.1, %18		; <i8>:124 [#uses=0]
	and i8 %vu8Mask_7.1, %18		; <i8>:125 [#uses=0]
	icmp eq i8 %104, %107		; <i1>:126 [#uses=1]
	br i1 %126, label %bb328, label %bb303

bb303:		; preds = %bb298
	call i16 @llvm.bswap.i16( i16 %81 )		; <i16>:127 [#uses=1]
	sub i16 %63, %77		; <i16>:128 [#uses=3]
	sub i16 %65, %79		; <i16>:129 [#uses=3]
	sub i16 %67, %73		; <i16>:130 [#uses=3]
	sub i16 %69, %75		; <i16>:131 [#uses=3]
	sub i16 %71, %127		; <i16>:132 [#uses=3]
	sub i16 0, %128		; <i16>:133 [#uses=1]
	icmp slt i16 %128, 0		; <i1>:134 [#uses=1]
	%.673 = select i1 %134, i16 %133, i16 %128		; <i16> [#uses=1]
	sub i16 0, %129		; <i16>:135 [#uses=1]
	icmp slt i16 %129, 0		; <i1>:136 [#uses=1]
	%.674 = select i1 %136, i16 %135, i16 %129		; <i16> [#uses=1]
	sub i16 0, %130		; <i16>:137 [#uses=1]
	icmp slt i16 %130, 0		; <i1>:138 [#uses=1]
	%.675 = select i1 %138, i16 %137, i16 %130		; <i16> [#uses=1]
	sub i16 0, %131		; <i16>:139 [#uses=1]
	icmp slt i16 %131, 0		; <i1>:140 [#uses=1]
	%.676 = select i1 %140, i16 %139, i16 %131		; <i16> [#uses=1]
	sub i16 0, %132		; <i16>:141 [#uses=1]
	icmp slt i16 %132, 0		; <i1>:142 [#uses=1]
	%.677 = select i1 %142, i16 %141, i16 %132		; <i16> [#uses=1]
	br label %bb328

bb328:		; preds = %bb303, %bb298
	%vu16Delta_0.0 = phi i16 [ %.673, %bb303 ], [ %., %bb298 ]		; <i16> [#uses=1]
	%vu16Delta_1.0 = phi i16 [ %.674, %bb303 ], [ %.660, %bb298 ]		; <i16> [#uses=0]
	%vu16Delta_2.0 = phi i16 [ %.675, %bb303 ], [ %.661, %bb298 ]		; <i16> [#uses=0]
	%vu16Delta_3.0 = phi i16 [ %.676, %bb303 ], [ %.662, %bb298 ]		; <i16> [#uses=0]
	%vu16Delta_6.0 = phi i16 [ %.677, %bb303 ], [ %.663, %bb298 ]		; <i16> [#uses=0]
	lshr i16 %vu16Delta_0.0, 8		; <i16>:143 [#uses=1]
	trunc i16 %143 to i8		; <i8>:144 [#uses=1]
	and i8 %144, %vu8Mask_0.1		; <i8>:145 [#uses=1]
	icmp eq i8 %145, 0		; <i1>:146 [#uses=0]
	sub i32 %105, %114		; <i32>:147 [#uses=1]
	sub i32 %111, %108		; <i32>:148 [#uses=1]
	or i32 %147, %148		; <i32>:149 [#uses=1]
	icmp eq i32 %149, 0		; <i1>:150 [#uses=0]
	call i32 @_Z54S_CalcIfLargeMVDeltaForBMbBothPredictionsFromSameFramePK19BiPartSrcDescriptorS1_ijj( ptr %57, ptr %59, i32 %19, i32 0, i32 0 ) nounwind 		; <i32>:151 [#uses=0]
	unreachable

labelContinueEdgesLoopV:		; preds = %bb206, %bb205, %bb144
	%fEdgeHasNonZeroBS.0 = phi i32 [ 0, %bb205 ], [ 0, %bb144 ], [ 1, %bb206 ]		; <i32> [#uses=2]
	%fMacroblockHasNonZeroBS.6 = phi i32 [ %152, %bb205 ], [ %fMacroblockHasNonZeroBS.4, %bb144 ], [ %152, %bb206 ]		; <i32> [#uses=1]
	%ixEdge.1 = phi i32 [ %160, %bb205 ], [ 0, %bb144 ], [ %160, %bb206 ]		; <i32> [#uses=1]
	%bfNZ12.2 = phi i32 [ %159, %bb205 ], [ 0, %bb144 ], [ %159, %bb206 ]		; <i32> [#uses=1]
	%boundaryStrengthsV.3 = phi ptr [ %158, %bb205 ], [ %boundaryStrengthsV.1771, %bb144 ], [ %158, %bb206 ]		; <ptr> [#uses=3]
	or i32 %fMacroblockHasNonZeroBS.6, %fEdgeHasNonZeroBS.0		; <i32>:152 [#uses=2]
	load i8, ptr %boundaryStrengthsV.3, align 1		; <i8>:153 [#uses=1]
	trunc i32 %fEdgeHasNonZeroBS.0 to i8		; <i8>:154 [#uses=1]
	shl i8 %154, 5		; <i8>:155 [#uses=1]
	xor i8 %155, 32		; <i8>:156 [#uses=1]
	or i8 %153, %156		; <i8>:157 [#uses=1]
	store i8 %157, ptr %boundaryStrengthsV.3, align 1
	getelementptr i8, ptr %boundaryStrengthsV.3, i32 4		; <ptr>:158 [#uses=4]
	shl i32 %bfNZ12.2, 4		; <i32>:159 [#uses=4]
	add i32 %ixEdge.1, 1		; <i32>:160 [#uses=6]
	icmp ult i32 %160, %numEdgesToTest.2		; <i1>:161 [#uses=1]
	br i1 %161, label %bb200, label %bb395

bb395:		; preds = %labelContinueEdgesLoopV
	add i32 %idxEachField11.0773, 1		; <i32>:162 [#uses=2]
	icmp ugt i32 %162, 0		; <i1>:163 [#uses=1]
	br i1 %163, label %bb398, label %bb144

bb398:		; preds = %bb395
	call void asm sideeffect "dcbt $0, $1", "b%,r,~{memory}"( i32 19, ptr null ) nounwind 
	unreachable

bb642:		; preds = %entry
	ret i32 0
}

declare i16 @llvm.bswap.i16(i16) nounwind readnone 

declare ptr @jvtNewPtrVectorAligned(i32)

declare ptr @jvtNewPtr(i32)

declare ptr @jvtNewPtrMemAligned(i32)

declare ptr @MutexNew()

declare ptr @_Znam(i32)

declare i32 @_Z24LoopFilter_FilterMbGroupP14LoopFilterInfoP11FramePixelsP22FrameMotionVectorCacheP19ThreadedBatchStructjjij(ptr, ptr, ptr, ptr, i32, i32, i32, i32)

declare void @MutexLock(ptr)

declare void @MutexUnlock(ptr)

declare i32 @_Z35LoopFilter_Internal_FilterLumaPlanePK14LoopFilterInfojjjjj(ptr, i32, i32, i32, i32, i32)

declare i32 @_Z37LoopFilter_Internal_FilterChromaPlanePK14LoopFilterInfojjjjj(ptr, i32, i32, i32, i32, i32)

declare void @_Z44LoopFilter_Internal_filter_macroblock_chromaPK14LoopFilterInfoPhS2_iiiPK30PerMacroblockBoundaryStrengthsjj(ptr, ptr, ptr, i32, i32, i32, ptr, i32, i32) nounwind 

declare i32 @_Z42LoopFilter_Internal_FilterChromaPlaneMBAFFPK14LoopFilterInfojjj(ptr, i32, i32, i32) nounwind 

declare i32 @_Z26LF_Threading2_ProcessTasksP14LoopFilterInfoP11FramePixelsP22FrameMotionVectorCacheij(ptr, ptr, ptr, i32, i32)

declare i32 @_Z46LoopFilter_Internal_CalculateBoundaryStrengthsPK14LoopFilterInfoP22FrameMotionVectorCachejj(ptr, ptr, i32, i32)

declare i32 @_Z44LoopFilter_Internal_FilterLumaChromaPlane_PPP14LoopFilterInfojjjjj(ptr, i32, i32, i32, i32, i32)

declare i32 @_Z22LoopFilter_FilterFrameP14LoopFilterInfoP11FramePixelsP22FrameMotionVectorCacheP19ThreadedBatchStructij(ptr, ptr, ptr, ptr, i32, i32)

declare void @_Z34LF_Threading2_ProcessTasks_WrapperPv(ptr)

