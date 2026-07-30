; RUN: llc -mtriple=arm-eabi %s -o - | FileCheck %s

	%struct.shape_edge_t = type { ptr, ptr, i32, i32, i32, i32 }
	%struct.shape_path_t = type { ptr, ptr, i32, i32, i32, i32, i32, i32 }
	%struct.shape_pool_t = type { ptr, ptr, ptr }

define ptr @shape_path_alloc(ptr %pool, ptr %shape) {
entry:
	br i1 false, label %cond_false, label %bb45

bb45:		; preds = %entry
	ret ptr null

cond_false:		; preds = %entry
	br i1 false, label %bb140, label %bb174

bb140:		; preds = %bb140, %cond_false
	%indvar = phi i32 [ 0, %cond_false ], [ %indvar.next, %bb140 ]		; <i32> [#uses=2]
	%edge.230.0.rec = shl i32 %indvar, 1		; <i32> [#uses=3]
	%edge.230.0 = getelementptr %struct.shape_edge_t, ptr null, i32 %edge.230.0.rec		; <ptr> [#uses=1]
	%edge.230.0.sum6970 = or i32 %edge.230.0.rec, 1		; <i32> [#uses=2]
	%tmp154 = getelementptr %struct.shape_edge_t, ptr null, i32 %edge.230.0.sum6970		; <ptr> [#uses=1]
	%tmp11.i5 = getelementptr %struct.shape_edge_t, ptr null, i32 %edge.230.0.sum6970, i32 0		; <ptr> [#uses=1]
	store ptr %edge.230.0, ptr %tmp11.i5
	store ptr %tmp154, ptr null
	%tmp16254.0.rec = add i32 %edge.230.0.rec, 2		; <i32> [#uses=1]
	%xp.350.sum = add i32 0, %tmp16254.0.rec		; <i32> [#uses=1]
	%tmp168 = icmp slt i32 %xp.350.sum, 0		; <i1> [#uses=1]
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=1]
	br i1 %tmp168, label %bb140, label %bb174

bb174:		; preds = %bb140, %cond_false
	ret ptr null
}

; CHECK-NOT: str{{.*}}!

