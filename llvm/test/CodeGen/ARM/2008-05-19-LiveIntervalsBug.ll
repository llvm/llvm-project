; RUN: llc < %s -mtriple=arm-apple-darwin

	%struct.BiContextType = type { i16, i8, i32 }
	%struct.Bitstream = type { i32, i32, i8, i32, i32, i8, i8, i32, i32, ptr, i32 }
	%struct.DataPartition = type { ptr, %struct.EncodingEnvironment, %struct.EncodingEnvironment }
	%struct.DecRefPicMarking_t = type { i32, i32, i32, i32, i32, ptr }
	%struct.EncodingEnvironment = type { i32, i32, i32, i32, i32, ptr, ptr, i32, i32 }
	%struct.ImageParameters = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, i32, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [9 x [16 x [16 x i16]]], [5 x [16 x [16 x i16]]], [9 x [8 x [8 x i16]]], [2 x [4 x [16 x [16 x i16]]]], [16 x [16 x i16]], [16 x [16 x i32]], ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, [4 x [4 x i32]], i32, i32, i32, i32, i32, double, i32, i32, i32, i32, ptr, ptr, ptr, ptr, [15 x i16], i32, i32, i32, i32, i32, i32, i32, i32, [6 x [32 x i32]], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [1 x i32], i32, i32, [2 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [3 x [2 x i32]], [2 x i32], i32, i32, i16, i32, i32, i32, i32, i32 }
	%struct.Macroblock = type { i32, i32, i32, [2 x i32], i32, [8 x i32], ptr, ptr, i32, [2 x [4 x [4 x [2 x i32]]]], [16 x i8], [16 x i8], i32, i64, [4 x i32], [4 x i32], i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i16, double, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.MotionInfoContexts = type { [3 x [11 x %struct.BiContextType]], [2 x [9 x %struct.BiContextType]], [2 x [10 x %struct.BiContextType]], [2 x [6 x %struct.BiContextType]], [4 x %struct.BiContextType], [4 x %struct.BiContextType], [3 x %struct.BiContextType] }
	%struct.Picture = type { i32, i32, [100 x ptr], i32, float, float, float }
	%struct.Slice = type { i32, i32, i32, i32, i32, i32, ptr, ptr, ptr, i32, ptr, ptr, ptr, i32, ptr, ptr, ptr, ptr, [3 x [2 x i32]] }
	%struct.TextureInfoContexts = type { [2 x %struct.BiContextType], [4 x %struct.BiContextType], [3 x [4 x %struct.BiContextType]], [10 x [4 x %struct.BiContextType]], [10 x [15 x %struct.BiContextType]], [10 x [15 x %struct.BiContextType]], [10 x [5 x %struct.BiContextType]], [10 x [5 x %struct.BiContextType]], [10 x [15 x %struct.BiContextType]], [10 x [15 x %struct.BiContextType]] }
@images = external global %struct.ImageParameters		; <ptr> [#uses=2]

declare ptr @calloc(i32, i32)

define fastcc void @init_global_buffers() nounwind {
entry:
	%tmp50.i.i = mul i32 0, 0		; <i32> [#uses=2]
	br i1 false, label %init_orig_buffers.exit, label %cond_true.i29

cond_true.i29:		; preds = %entry
	%tmp17.i = load i32, ptr getelementptr (%struct.ImageParameters, ptr @images, i32 0, i32 20), align 8		; <i32> [#uses=1]
	%tmp20.i27 = load i32, ptr getelementptr (%struct.ImageParameters, ptr @images, i32 0, i32 16), align 8		; <i32> [#uses=1]
	%tmp8.i.i = select i1 false, i32 1, i32 0		; <i32> [#uses=1]
	br label %bb.i8.us.i

bb.i8.us.i:		; preds = %get_mem2Dpel.exit.i.us.i, %cond_true.i29
	%j.04.i.us.i = phi i32 [ %indvar.next39.i, %get_mem2Dpel.exit.i.us.i ], [ 0, %cond_true.i29 ]		; <i32> [#uses=2]
	%tmp13.i.us.i = getelementptr ptr, ptr null, i32 %j.04.i.us.i		; <ptr> [#uses=0]
	%tmp15.i.i.us.i = tail call ptr @calloc( i32 0, i32 2 )		; <ptr> [#uses=0]
	store ptr null, ptr null, align 4
	br label %bb.i.i.us.i

get_mem2Dpel.exit.i.us.i:		; preds = %bb.i.i.us.i
	%indvar.next39.i = add i32 %j.04.i.us.i, 1		; <i32> [#uses=2]
	%exitcond40.i = icmp eq i32 %indvar.next39.i, 2		; <i1> [#uses=1]
	br i1 %exitcond40.i, label %get_mem3Dpel.exit.split.i, label %bb.i8.us.i

bb.i.i.us.i:		; preds = %bb.i.i.us.i, %bb.i8.us.i
	%exitcond.i = icmp eq i32 0, %tmp8.i.i		; <i1> [#uses=1]
	br i1 %exitcond.i, label %get_mem2Dpel.exit.i.us.i, label %bb.i.i.us.i

get_mem3Dpel.exit.split.i:		; preds = %get_mem2Dpel.exit.i.us.i
	%tmp30.i.i = shl i32 %tmp17.i, 2		; <i32> [#uses=1]
	%tmp31.i.i = mul i32 %tmp30.i.i, %tmp20.i27		; <i32> [#uses=1]
	%tmp23.i31 = add i32 %tmp31.i.i, %tmp50.i.i		; <i32> [#uses=1]
	br label %init_orig_buffers.exit

init_orig_buffers.exit:		; preds = %get_mem3Dpel.exit.split.i, %entry
	%memory_size.0.i = phi i32 [ %tmp23.i31, %get_mem3Dpel.exit.split.i ], [ %tmp50.i.i, %entry ]		; <i32> [#uses=1]
	%tmp41 = add i32 0, %memory_size.0.i		; <i32> [#uses=0]
	unreachable
}
