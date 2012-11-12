; RUN: llc < %s -mtriple=x86_64-linux-gnu | grep addb | not grep x
; RUN: llc < %s -mtriple=x86_64-linux-gnu | grep cmpb | not grep x
; PR1734

target triple = "x86_64-unknown-linux-gnu"
	%struct.CUMULATIVE_ARGS = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.eh_status = type opaque
	%struct.emit_status = type { i32, i32, %struct.rtx_def*, %struct.rtx_def*, %struct.sequence_stack*, i32, %struct.location_t, i32, i8*, %struct.rtx_def** }
	%struct.expr_status = type { i32, i32, i32, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def* }
	%struct.function = type { %struct.eh_status*, %struct.expr_status*, %struct.emit_status*, %struct.varasm_status*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.function*, i32, i32, i32, i32, %struct.rtx_def*, %struct.CUMULATIVE_ARGS, %struct.rtx_def*, %struct.rtx_def*, %struct.initial_value_struct*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, i8, i32, i64, %struct.tree_node*, %struct.tree_node*, %struct.rtx_def*, %struct.varray_head_tag*, %struct.temp_slot*, i32, %struct.var_refs_queue*, i32, i32, %struct.rtvec_def*, %struct.tree_node*, i32, i32, i32, %struct.machine_function*, i32, i32, i8, i8, %struct.language_function*, %struct.rtx_def*, i32, i32, i32, i32, %struct.location_t, %struct.varray_head_tag*, %struct.tree_node*, %struct.tree_node*, i8, i8, i8 }
	%struct.initial_value_struct = type opaque
	%struct.lang_decl = type opaque
	%struct.language_function = type opaque
	%struct.location_t = type { i8*, i32 }
	%struct.machine_function = type { %struct.stack_local_entry*, i8*, %struct.rtx_def*, i32, i32, i32, i32, i32 }
	%struct.rtunion = type { i8* }
	%struct.rtvec_def = type { i32, [1 x %struct.rtx_def*] }
	%struct.rtx_def = type { i16, i8, i8, %struct.u }
	%struct.sequence_stack = type { %struct.rtx_def*, %struct.rtx_def*, %struct.sequence_stack* }
	%struct.stack_local_entry = type opaque
	%struct.temp_slot = type opaque
	%struct.tree_common = type { %struct.tree_node*, %struct.tree_node*, %union.tree_ann_d*, i8, i8, i8, i8, i8 }
	%struct.tree_decl = type { %struct.tree_common, %struct.location_t, i32, %struct.tree_node*, i8, i8, i8, i8, i8, i8, i8, i8, i32, %struct.tree_decl_u1, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.rtx_def*, i32, %struct.tree_decl_u2, %struct.tree_node*, %struct.tree_node*, i64, %struct.lang_decl* }
	%struct.tree_decl_u1 = type { i64 }
	%struct.tree_decl_u2 = type { %struct.function* }
	%struct.tree_node = type { %struct.tree_decl }
	%struct.u = type { [1 x %struct.rtunion] }
	%struct.var_refs_queue = type { %struct.rtx_def*, i32, i32, %struct.var_refs_queue* }
	%struct.varasm_status = type opaque
	%struct.varray_data = type { [1 x i64] }
	%struct.varray_head_tag = type { i64, i64, i32, i8*, %struct.varray_data }
	%union.tree_ann_d = type opaque

define void @layout_type(%struct.tree_node* %type) {
entry:
	%tmp32 = load i32* null, align 8		; <i32> [#uses=3]
	%tmp3435 = trunc i32 %tmp32 to i8		; <i8> [#uses=1]
	%tmp53 = icmp eq %struct.tree_node* null, null		; <i1> [#uses=1]
	br i1 %tmp53, label %cond_next57, label %UnifiedReturnBlock

cond_next57:		; preds = %entry
	%tmp65 = and i32 %tmp32, 255		; <i32> [#uses=1]
	switch i32 %tmp65, label %UnifiedReturnBlock [
		 i32 6, label %bb140
		 i32 7, label %bb140
		 i32 8, label %bb140
		 i32 13, label %bb478
	]

bb140:		; preds = %cond_next57, %cond_next57, %cond_next57
	%tmp219 = load i32* null, align 8		; <i32> [#uses=1]
	%tmp221222 = trunc i32 %tmp219 to i8		; <i8> [#uses=1]
	%tmp223 = icmp eq i8 %tmp221222, 24		; <i1> [#uses=1]
	br i1 %tmp223, label %cond_true226, label %cond_next340

cond_true226:		; preds = %bb140
	switch i8 %tmp3435, label %cond_true288 [
		 i8 6, label %cond_next340
		 i8 9, label %cond_next340
		 i8 7, label %cond_next340
		 i8 8, label %cond_next340
		 i8 10, label %cond_next340
	]

cond_true288:		; preds = %cond_true226
	unreachable

cond_next340:		; preds = %cond_true226, %cond_true226, %cond_true226, %cond_true226, %cond_true226, %bb140
	ret void

bb478:		; preds = %cond_next57
	br i1 false, label %cond_next500, label %cond_true497

cond_true497:		; preds = %bb478
	unreachable

cond_next500:		; preds = %bb478
	%tmp513 = load i32* null, align 8		; <i32> [#uses=1]
	%tmp545 = and i32 %tmp513, 8192		; <i32> [#uses=1]
	%tmp547 = and i32 %tmp32, -8193		; <i32> [#uses=1]
	%tmp548 = or i32 %tmp547, %tmp545		; <i32> [#uses=1]
	store i32 %tmp548, i32* null, align 8
	ret void

UnifiedReturnBlock:		; preds = %cond_next57, %entry
	ret void
}
