; RUN: opt < %s -passes=indvars -S | FileCheck %s
; ModuleID = '<stdin>'

; Provide legal integer types.
target datalayout = "n8:16:32:64"


	%struct.App1Marker = type <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }>
	%struct.ComponentInstanceRecord = type <{ [1 x i32] }>
	%struct.DCPredictors = type { [5 x i16] }
	%struct.DecodeTable = type { i16, i16, i16, i16, ptr, ptr }
	%struct.ICMDataProcRecord = type <{ ptr, i32 }>
	%struct.JPEGBitStream = type { ptr, i32, i32, i32, i32, i32, ptr, ptr, i32, i16, i16, i32 }
	%struct.JPEGGlobals = type { [2048 x i8], %struct.JPEGBitStream, ptr, i32, i32, ptr, ptr, i32, ptr, %struct.Rect, i32, i32, %struct.SharedGlobals, %struct.DCPredictors, i8, i8, ptr, %struct.YUVGeneralParams, i16, i16, i32, [5 x ptr], [5 x ptr], [5 x ptr], [5 x i8], [5 x i8], [4 x [65 x i16]], [4 x %struct.DecodeTable], [4 x %struct.DecodeTable], [4 x ptr], [4 x ptr], i16, i16, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, [18 x i8], [18 x i8], [18 x i8], [18 x i8], i32, i32, ptr, ptr, i8, i8, i8, i8, i16, i16, ptr, i8, i8, i8, i8, ptr, ptr, ptr, ptr, ptr, i8, [3 x i8], i32, [3 x i32], [3 x i32], [3 x i32], [3 x i32], [3 x i32], [3 x ptr], [3 x ptr], [3 x ptr], [3 x ptr], [3 x ptr], [3 x i32], i32, [3 x ptr], i32, i32, i32, [3 x i32], i8, i8, i8, i8, ptr, i32, i32, ptr, ptr, ptr, ptr, i32, i32, ptr, i32, i32, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, [16 x <2 x i64>], [1280 x i8], i8 }
	%struct.OpaqueQTMLMutex = type opaque
	%struct.Rect = type { i16, i16, i16, i16 }
	%struct.SharedDGlobals = type { %struct.DecodeTable, %struct.DecodeTable, %struct.DecodeTable, %struct.DecodeTable }
	%struct.SharedEGlobals = type { ptr, ptr, ptr, ptr }
	%struct.SharedGlobals = type { ptr, ptr }
	%struct.YUVGeneralParams = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, i16, i16, i16, [6 x i8], ptr, i16, i16 }
@llvm.used = appending global [1 x ptr] [ ptr @ExtractBufferedBlocksIgnored ], section "llvm.metadata"		; <ptr> [#uses=0]

define signext i16 @ExtractBufferedBlocksIgnored(ptr %globp)  nounwind {
; CHECK-LABEL: @ExtractBufferedBlocksIgnored(
; CHECK: sext
; CHECK-NOT: sext
entry:
	%tmp4311 = getelementptr %struct.JPEGGlobals, ptr %globp, i32 0, i32 70		; <ptr> [#uses=1]
	%tmp4412 = load i32, ptr %tmp4311, align 16		; <i32> [#uses=2]
	%tmp4613 = icmp sgt i32 %tmp4412, 0		; <i1> [#uses=1]
	br i1 %tmp4613, label %bb, label %bb49

bb:		; preds = %bb28, %entry
	%component.09 = phi i16 [ 0, %entry ], [ %tmp37, %bb28 ]		; <i16> [#uses=2]
	%tmp12 = sext i16 %component.09 to i32		; <i32> [#uses=2]
	%tmp6 = getelementptr %struct.JPEGGlobals, ptr %globp, i32 0, i32 77, i32 %tmp12		; <ptr> [#uses=2]
	%tmp7 = load ptr, ptr %tmp6, align 4		; <ptr> [#uses=2]
	%tmp235 = getelementptr %struct.JPEGGlobals, ptr %globp, i32 0, i32 71, i32 %tmp12		; <ptr> [#uses=1]
	%tmp246 = load i32, ptr %tmp235, align 4		; <i32> [#uses=2]
	%tmp267 = icmp sgt i32 %tmp246, 0		; <i1> [#uses=1]
	br i1 %tmp267, label %bb8, label %bb28

bb8:		; preds = %bb8, %bb
	%indvar = phi i32 [ 0, %bb ], [ %indvar.next2, %bb8 ]		; <i32> [#uses=3]
	%theDCTBufferIter.01.rec = shl i32 %indvar, 6		; <i32> [#uses=1]
	%tmp10.rec = add i32 %theDCTBufferIter.01.rec, 64		; <i32> [#uses=1]
	%tmp10 = getelementptr i16, ptr %tmp7, i32 %tmp10.rec		; <ptr> [#uses=1]
	%i.02 = trunc i32 %indvar to i16		; <i16> [#uses=1]
	%tmp13 = add i16 %i.02, 1		; <i16> [#uses=1]
	%phitmp = sext i16 %tmp13 to i32		; <i32> [#uses=1]
	%tmp26 = icmp slt i32 %phitmp, %tmp246		; <i1> [#uses=1]
	%indvar.next2 = add i32 %indvar, 1		; <i32> [#uses=1]
	br i1 %tmp26, label %bb8, label %bb28

bb28:		; preds = %bb8, %bb
	%theDCTBufferIter.0.lcssa = phi ptr [ %tmp7, %bb ], [ %tmp10, %bb8 ]		; <ptr> [#uses=1]
	store ptr %theDCTBufferIter.0.lcssa, ptr %tmp6, align 4
	%tmp37 = add i16 %component.09, 1		; <i16> [#uses=2]
	%phitmp15 = sext i16 %tmp37 to i32		; <i32> [#uses=1]
	%tmp46 = icmp slt i32 %phitmp15, 42		; <i1> [#uses=1]
	br i1 %tmp46, label %bb, label %bb49

bb49:		; preds = %bb28, %entry
	ret i16 0
}
