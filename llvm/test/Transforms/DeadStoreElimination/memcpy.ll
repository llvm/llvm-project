; RUN: llvm-as < %s | opt -dse | llvm-dis | not grep alloca
; ModuleID = 'placeholder.adb'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i686-pc-linux-gnu"
	%struct.placeholder__T5b = type { i32, [1 x i32] }
	%struct.placeholder__an_interval___PAD = type { %struct.placeholder__interval, [4 x i32] }
	%struct.placeholder__interval = type { i32, i32 }
	%struct.placeholder__s__s__the_interval___PAD = type { %struct.placeholder__interval }

define void @_ada_placeholder() nounwind  {
entry:
	%an_interval = alloca %struct.placeholder__an_interval___PAD		; <%struct.placeholder__an_interval___PAD*> [#uses=3]
	%tmp34 = bitcast %struct.placeholder__an_interval___PAD* %an_interval to %struct.placeholder__T5b*		; <%struct.placeholder__T5b*> [#uses=1]
	%tmp5 = getelementptr %struct.placeholder__an_interval___PAD* %an_interval, i32 0, i32 0, i32 0		; <i32*> [#uses=2]
	store i32 1, i32* %tmp5, align 8
	%tmp10 = getelementptr %struct.placeholder__T5b* %tmp34, i32 0, i32 1, i32 0		; <i32*> [#uses=1]
	store i32 1, i32* %tmp10, align 4
	%tmp82 = load i32* %tmp5, align 8		; <i32> [#uses=5]
	%tmp83 = icmp slt i32 %tmp82, 6		; <i1> [#uses=1]
	%min84 = select i1 %tmp83, i32 %tmp82, i32 5		; <i32> [#uses=3]
	%tmp85 = icmp sgt i32 %min84, -1		; <i1> [#uses=2]
	%min84.cast193 = zext i32 %min84 to i64		; <i64> [#uses=1]
	%min84.cast193.op = shl i64 %min84.cast193, 33		; <i64> [#uses=1]
	%tmp104 = icmp sgt i32 %tmp82, -1		; <i1> [#uses=2]
	%tmp103.cast192 = zext i32 %tmp82 to i64		; <i64> [#uses=1]
	%tmp103.cast192.op = shl i64 %tmp103.cast192, 33		; <i64> [#uses=1]
	%min84.cast193.op.op = ashr i64 %min84.cast193.op, 28		; <i64> [#uses=1]
	%sextr121 = select i1 %tmp85, i64 %min84.cast193.op.op, i64 0		; <i64> [#uses=2]
	%tmp103.cast192.op.op = ashr i64 %tmp103.cast192.op, 28		; <i64> [#uses=1]
	%sextr123 = select i1 %tmp104, i64 %tmp103.cast192.op.op, i64 0		; <i64> [#uses=2]
	%tmp124 = icmp sle i64 %sextr121, %sextr123		; <i1> [#uses=1]
	%min125 = select i1 %tmp124, i64 %sextr121, i64 %sextr123		; <i64> [#uses=1]
	%sextr131194 = and i64 %min125, 34359738336		; <i64> [#uses=1]
	%tmp134 = add i64 %sextr131194, 63		; <i64> [#uses=1]
	lshr i64 %tmp134, 3		; <i64>:0 [#uses=1]
	%tmp150188.shrunk = trunc i64 %0 to i32		; <i32> [#uses=1]
	%tmp159 = and i32 %tmp150188.shrunk, -4		; <i32> [#uses=1]
	%tmp161 = alloca i8, i32 %tmp159		; <i8*> [#uses=1]
	%min167.op = shl i32 %min84, 2		; <i32> [#uses=1]
	%tmp170 = select i1 %tmp85, i32 %min167.op, i32 0		; <i32> [#uses=2]
	%tmp173.op = shl i32 %tmp82, 2		; <i32> [#uses=1]
	%tmp176 = select i1 %tmp104, i32 %tmp173.op, i32 0		; <i32> [#uses=2]
	%tmp177 = icmp sle i32 %tmp170, %tmp176		; <i1> [#uses=1]
	%min178 = select i1 %tmp177, i32 %tmp170, i32 %tmp176		; <i32> [#uses=1]
	%tmp179 = add i32 %min178, 7		; <i32> [#uses=1]
	%tmp180 = and i32 %tmp179, -4		; <i32> [#uses=1]
	%tmp183185 = bitcast %struct.placeholder__an_interval___PAD* %an_interval to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32( i8* %tmp161, i8* %tmp183185, i32 %tmp180, i32 4 )
	ret void
}

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32) nounwind 
