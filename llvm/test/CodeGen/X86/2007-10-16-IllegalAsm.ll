; RUN: llvm-as < %s | llc -mtriple=x86_64-linux-gnu | grep movb | not grep x
; PR1734

	%struct.CUMULATIVE_ARGS = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.eh_status = type opaque
	%struct.emit_status = type { i32, i32, %struct.rtx_def*, %struct.rtx_def*, %struct.sequence_stack*, i32, %struct.location_t, i32, i8*, %struct.rtx_def** }
	%struct.expr_status = type { i32, i32, i32, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def* }
	%struct.function = type { %struct.eh_status*, %struct.expr_status*, %struct.emit_status*, %struct.varasm_status*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.function*, i32, i32, i32, i32, %struct.rtx_def*, %struct.CUMULATIVE_ARGS, %struct.rtx_def*, %struct.rtx_def*, %struct.initial_value_struct*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, i8, i32, i64, %struct.tree_node*, %struct.tree_node*, %struct.rtx_def*, %struct.varray_head_tag*, %struct.temp_slot*, i32, %struct.var_refs_queue*, i32, i32, %struct.rtvec_def*, %struct.tree_node*, i32, i32, i32, %struct.machine_function*, i32, i32, i8, i8, %struct.language_function*, %struct.rtx_def*, i32, i32, i32, i32, %struct.location_t, %struct.varray_head_tag*, %struct.tree_node*, %struct.tree_node*, i8, i8, i8 }
	%struct.initial_value_struct = type opaque
	%struct.lang_decl = type opaque
	%struct.lang_type = type opaque
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
	%struct.tree_type = type { %struct.tree_common, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, i32, i16, i8, i8, i32, %struct.tree_node*, %struct.tree_node*, %struct.rtunion, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, i64, %struct.lang_type* }
	%struct.u = type { [1 x %struct.rtunion] }
	%struct.var_refs_queue = type { %struct.rtx_def*, i32, i32, %struct.var_refs_queue* }
	%struct.varasm_status = type opaque
	%struct.varray_data = type { [1 x i64] }
	%struct.varray_head_tag = type { i64, i64, i32, i8*, %struct.varray_data }
	%union.tree_ann_d = type opaque
@.str = external constant [28 x i8]		; <[28 x i8]*> [#uses=1]
@tree_code_type = external constant [0 x i32]		; <[0 x i32]*> [#uses=5]
@global_trees = external global [47 x %struct.tree_node*]		; <[47 x %struct.tree_node*]*> [#uses=1]
@mode_size = external global [48 x i8]		; <[48 x i8]*> [#uses=1]
@__FUNCTION__.22683 = external constant [12 x i8]		; <[12 x i8]*> [#uses=1]

define void @layout_type(%struct.tree_node* %type) {
entry:
	%tmp15 = icmp eq %struct.tree_node* %type, null		; <i1> [#uses=1]
	br i1 %tmp15, label %cond_true, label %cond_false

cond_true:		; preds = %entry
	tail call void @fancy_abort( i8* getelementptr ([28 x i8]* @.str, i32 0, i64 0), i32 1713, i8* getelementptr ([12 x i8]* @__FUNCTION__.22683, i32 0, i32 0) )
	unreachable

cond_false:		; preds = %entry
	%tmp19 = load %struct.tree_node** getelementptr ([47 x %struct.tree_node*]* @global_trees, i32 0, i64 0), align 8		; <%struct.tree_node*> [#uses=1]
	%tmp21 = icmp eq %struct.tree_node* %tmp19, %type		; <i1> [#uses=1]
	br i1 %tmp21, label %UnifiedReturnBlock, label %cond_next25

cond_next25:		; preds = %cond_false
	%tmp30 = getelementptr %struct.tree_node* %type, i32 0, i32 0, i32 0, i32 3		; <i8*> [#uses=1]
	%tmp3031 = bitcast i8* %tmp30 to i32*		; <i32*> [#uses=6]
	%tmp32 = load i32* %tmp3031, align 8		; <i32> [#uses=3]
	%tmp3435 = trunc i32 %tmp32 to i8		; <i8> [#uses=3]
	%tmp34353637 = zext i8 %tmp3435 to i64		; <i64> [#uses=1]
	%tmp38 = getelementptr [0 x i32]* @tree_code_type, i32 0, i64 %tmp34353637		; <i32*> [#uses=1]
	%tmp39 = load i32* %tmp38, align 4		; <i32> [#uses=1]
	%tmp40 = icmp eq i32 %tmp39, 2		; <i1> [#uses=4]
	br i1 %tmp40, label %cond_next46, label %cond_true43

cond_true43:		; preds = %cond_next25
	tail call void @tree_class_check_failed( %struct.tree_node* %type, i32 2, i8* getelementptr ([28 x i8]* @.str, i32 0, i64 0), i32 1719, i8* getelementptr ([12 x i8]* @__FUNCTION__.22683, i32 0, i32 0) )
	unreachable

cond_next46:		; preds = %cond_next25
	%tmp4950 = bitcast %struct.tree_node* %type to %struct.tree_type*		; <%struct.tree_type*> [#uses=2]
	%tmp51 = getelementptr %struct.tree_type* %tmp4950, i32 0, i32 2		; <%struct.tree_node**> [#uses=2]
	%tmp52 = load %struct.tree_node** %tmp51, align 8		; <%struct.tree_node*> [#uses=1]
	%tmp53 = icmp eq %struct.tree_node* %tmp52, null		; <i1> [#uses=1]
	br i1 %tmp53, label %cond_next57, label %UnifiedReturnBlock

cond_next57:		; preds = %cond_next46
	%tmp65 = and i32 %tmp32, 255		; <i32> [#uses=1]
	switch i32 %tmp65, label %UnifiedReturnBlock [
		 i32 6, label %bb140
		 i32 7, label %bb69
		 i32 8, label %bb140
		 i32 13, label %bb478
		 i32 23, label %bb
	]

bb:		; preds = %cond_next57
	tail call void @fancy_abort( i8* getelementptr ([28 x i8]* @.str, i32 0, i64 0), i32 1727, i8* getelementptr ([12 x i8]* @__FUNCTION__.22683, i32 0, i32 0) )
	unreachable

bb69:		; preds = %cond_next57
	br i1 %tmp40, label %cond_next91, label %cond_true88

cond_true88:		; preds = %bb69
	tail call void @tree_class_check_failed( %struct.tree_node* %type, i32 2, i8* getelementptr ([28 x i8]* @.str, i32 0, i64 0), i32 1730, i8* getelementptr ([12 x i8]* @__FUNCTION__.22683, i32 0, i32 0) )
	unreachable

cond_next91:		; preds = %bb69
	%tmp96 = getelementptr %struct.tree_node* %type, i32 0, i32 0, i32 8		; <i8*> [#uses=1]
	%tmp9697 = bitcast i8* %tmp96 to i32*		; <i32*> [#uses=2]
	%tmp98 = load i32* %tmp9697, align 8		; <i32> [#uses=2]
	%tmp100101552 = and i32 %tmp98, 511		; <i32> [#uses=1]
	%tmp102 = icmp eq i32 %tmp100101552, 0		; <i1> [#uses=1]
	br i1 %tmp102, label %cond_true105, label %bb140

cond_true105:		; preds = %cond_next91
	br i1 %tmp40, label %cond_next127, label %cond_true124

cond_true124:		; preds = %cond_true105
	tail call void @tree_class_check_failed( %struct.tree_node* %type, i32 2, i8* getelementptr ([28 x i8]* @.str, i32 0, i64 0), i32 1731, i8* getelementptr ([12 x i8]* @__FUNCTION__.22683, i32 0, i32 0) )
	unreachable

cond_next127:		; preds = %cond_true105
	%tmp136 = or i32 %tmp98, 1		; <i32> [#uses=1]
	%tmp137 = and i32 %tmp136, -511		; <i32> [#uses=1]
	store i32 %tmp137, i32* %tmp9697, align 8
	br label %bb140

bb140:		; preds = %cond_next127, %cond_next91, %cond_next57, %cond_next57
	switch i8 %tmp3435, label %cond_true202 [
		 i8 6, label %cond_next208
		 i8 9, label %cond_next208
		 i8 7, label %cond_next208
		 i8 8, label %cond_next208
		 i8 10, label %cond_next208
	]

cond_true202:		; preds = %bb140
	tail call void (%struct.tree_node*, i8*, i32, i8*, ...)* @tree_check_failed( %struct.tree_node* %type, i8* getelementptr ([28 x i8]* @.str, i32 0, i64 0), i32 1738, i8* getelementptr ([12 x i8]* @__FUNCTION__.22683, i32 0, i32 0), i32 9, i32 6, i32 7, i32 8, i32 10, i32 0 )
	unreachable

cond_next208:		; preds = %bb140, %bb140, %bb140, %bb140, %bb140
	%tmp213 = getelementptr %struct.tree_type* %tmp4950, i32 0, i32 14		; <%struct.tree_node**> [#uses=1]
	%tmp214 = load %struct.tree_node** %tmp213, align 8		; <%struct.tree_node*> [#uses=2]
	%tmp217 = getelementptr %struct.tree_node* %tmp214, i32 0, i32 0, i32 0, i32 3		; <i8*> [#uses=1]
	%tmp217218 = bitcast i8* %tmp217 to i32*		; <i32*> [#uses=1]
	%tmp219 = load i32* %tmp217218, align 8		; <i32> [#uses=1]
	%tmp221222 = trunc i32 %tmp219 to i8		; <i8> [#uses=1]
	%tmp223 = icmp eq i8 %tmp221222, 24		; <i1> [#uses=1]
	br i1 %tmp223, label %cond_true226, label %cond_next340

cond_true226:		; preds = %cond_next208
	switch i8 %tmp3435, label %cond_true288 [
		 i8 6, label %cond_next294
		 i8 9, label %cond_next294
		 i8 7, label %cond_next294
		 i8 8, label %cond_next294
		 i8 10, label %cond_next294
	]

cond_true288:		; preds = %cond_true226
	tail call void (%struct.tree_node*, i8*, i32, i8*, ...)* @tree_check_failed( %struct.tree_node* %type, i8* getelementptr ([28 x i8]* @.str, i32 0, i64 0), i32 1739, i8* getelementptr ([12 x i8]* @__FUNCTION__.22683, i32 0, i32 0), i32 9, i32 6, i32 7, i32 8, i32 10, i32 0 )
	unreachable

cond_next294:		; preds = %cond_true226, %cond_true226, %cond_true226, %cond_true226, %cond_true226
	%tmp301 = tail call i32 @tree_int_cst_sgn( %struct.tree_node* %tmp214 )		; <i32> [#uses=1]
	%tmp302 = icmp sgt i32 %tmp301, -1		; <i1> [#uses=1]
	br i1 %tmp302, label %cond_true305, label %cond_next340

cond_true305:		; preds = %cond_next294
	%tmp313 = load i32* %tmp3031, align 8		; <i32> [#uses=2]
	%tmp315316 = trunc i32 %tmp313 to i8		; <i8> [#uses=1]
	%tmp315316317318 = zext i8 %tmp315316 to i64		; <i64> [#uses=1]
	%tmp319 = getelementptr [0 x i32]* @tree_code_type, i32 0, i64 %tmp315316317318		; <i32*> [#uses=1]
	%tmp320 = load i32* %tmp319, align 4		; <i32> [#uses=1]
	%tmp321 = icmp eq i32 %tmp320, 2		; <i1> [#uses=1]
	br i1 %tmp321, label %cond_next327, label %cond_true324

cond_true324:		; preds = %cond_true305
	tail call void @tree_class_check_failed( %struct.tree_node* %type, i32 2, i8* getelementptr ([28 x i8]* @.str, i32 0, i64 0), i32 1740, i8* getelementptr ([12 x i8]* @__FUNCTION__.22683, i32 0, i32 0) )
	unreachable

cond_next327:		; preds = %cond_true305
	%tmp338 = or i32 %tmp313, 8192		; <i32> [#uses=1]
	store i32 %tmp338, i32* %tmp3031, align 8
	br label %cond_next340

cond_next340:		; preds = %cond_next327, %cond_next294, %cond_next208
	%tmp348 = load i32* %tmp3031, align 8		; <i32> [#uses=1]
	%tmp350351 = trunc i32 %tmp348 to i8		; <i8> [#uses=1]
	%tmp350351352353 = zext i8 %tmp350351 to i64		; <i64> [#uses=1]
	%tmp354 = getelementptr [0 x i32]* @tree_code_type, i32 0, i64 %tmp350351352353		; <i32*> [#uses=1]
	%tmp355 = load i32* %tmp354, align 4		; <i32> [#uses=1]
	%tmp356 = icmp eq i32 %tmp355, 2		; <i1> [#uses=1]
	br i1 %tmp356, label %cond_next385, label %cond_true359

cond_true359:		; preds = %cond_next340
	tail call void @tree_class_check_failed( %struct.tree_node* %type, i32 2, i8* getelementptr ([28 x i8]* @.str, i32 0, i64 0), i32 1742, i8* getelementptr ([12 x i8]* @__FUNCTION__.22683, i32 0, i32 0) )
	unreachable

cond_next385:		; preds = %cond_next340
	%tmp390 = getelementptr %struct.tree_node* %type, i32 0, i32 0, i32 8		; <i8*> [#uses=1]
	%tmp390391 = bitcast i8* %tmp390 to i32*		; <i32*> [#uses=3]
	%tmp392 = load i32* %tmp390391, align 8		; <i32> [#uses=1]
	%tmp394 = and i32 %tmp392, 511		; <i32> [#uses=1]
	%tmp397 = tail call i32 @smallest_mode_for_size( i32 %tmp394, i32 2 )		; <i32> [#uses=1]
	%tmp404 = load i32* %tmp390391, align 8		; <i32> [#uses=1]
	%tmp397398405 = shl i32 %tmp397, 9		; <i32> [#uses=1]
	%tmp407 = and i32 %tmp397398405, 65024		; <i32> [#uses=1]
	%tmp408 = and i32 %tmp404, -65025		; <i32> [#uses=1]
	%tmp409 = or i32 %tmp408, %tmp407		; <i32> [#uses=2]
	store i32 %tmp409, i32* %tmp390391, align 8
	%tmp417 = load i32* %tmp3031, align 8		; <i32> [#uses=1]
	%tmp419420 = trunc i32 %tmp417 to i8		; <i8> [#uses=1]
	%tmp419420421422 = zext i8 %tmp419420 to i64		; <i64> [#uses=1]
	%tmp423 = getelementptr [0 x i32]* @tree_code_type, i32 0, i64 %tmp419420421422		; <i32*> [#uses=1]
	%tmp424 = load i32* %tmp423, align 4		; <i32> [#uses=1]
	%tmp425 = icmp eq i32 %tmp424, 2		; <i1> [#uses=1]
	br i1 %tmp425, label %cond_next454, label %cond_true428

cond_true428:		; preds = %cond_next385
	tail call void @tree_class_check_failed( %struct.tree_node* %type, i32 2, i8* getelementptr ([28 x i8]* @.str, i32 0, i64 0), i32 1744, i8* getelementptr ([12 x i8]* @__FUNCTION__.22683, i32 0, i32 0) )
	unreachable

cond_next454:		; preds = %cond_next385
	lshr i32 %tmp409, 9		; <i32>:0 [#uses=1]
	trunc i32 %0 to i8		; <i8>:1 [#uses=1]
	%tmp463464 = and i8 %1, 127		; <i8> [#uses=1]
	%tmp463464465466 = zext i8 %tmp463464 to i64		; <i64> [#uses=1]
	%tmp467 = getelementptr [48 x i8]* @mode_size, i32 0, i64 %tmp463464465466		; <i8*> [#uses=1]
	%tmp468 = load i8* %tmp467, align 1		; <i8> [#uses=1]
	%tmp468469553 = zext i8 %tmp468 to i16		; <i16> [#uses=1]
	%tmp470471 = shl i16 %tmp468469553, 3		; <i16> [#uses=1]
	%tmp470471472 = zext i16 %tmp470471 to i64		; <i64> [#uses=1]
	%tmp473 = tail call %struct.tree_node* @size_int_kind( i64 %tmp470471472, i32 2 )		; <%struct.tree_node*> [#uses=1]
	store %struct.tree_node* %tmp473, %struct.tree_node** %tmp51, align 8
	ret void

bb478:		; preds = %cond_next57
	br i1 %tmp40, label %cond_next500, label %cond_true497

cond_true497:		; preds = %bb478
	tail call void @tree_class_check_failed( %struct.tree_node* %type, i32 2, i8* getelementptr ([28 x i8]* @.str, i32 0, i64 0), i32 1755, i8* getelementptr ([12 x i8]* @__FUNCTION__.22683, i32 0, i32 0) )
	unreachable

cond_next500:		; preds = %bb478
	%tmp506 = getelementptr %struct.tree_node* %type, i32 0, i32 0, i32 0, i32 1		; <%struct.tree_node**> [#uses=1]
	%tmp507 = load %struct.tree_node** %tmp506, align 8		; <%struct.tree_node*> [#uses=2]
	%tmp511 = getelementptr %struct.tree_node* %tmp507, i32 0, i32 0, i32 0, i32 3		; <i8*> [#uses=1]
	%tmp511512 = bitcast i8* %tmp511 to i32*		; <i32*> [#uses=1]
	%tmp513 = load i32* %tmp511512, align 8		; <i32> [#uses=2]
	%tmp515516 = trunc i32 %tmp513 to i8		; <i8> [#uses=1]
	%tmp515516517518 = zext i8 %tmp515516 to i64		; <i64> [#uses=1]
	%tmp519 = getelementptr [0 x i32]* @tree_code_type, i32 0, i64 %tmp515516517518		; <i32*> [#uses=1]
	%tmp520 = load i32* %tmp519, align 4		; <i32> [#uses=1]
	%tmp521 = icmp eq i32 %tmp520, 2		; <i1> [#uses=1]
	br i1 %tmp521, label %cond_next527, label %cond_true524

cond_true524:		; preds = %cond_next500
	tail call void @tree_class_check_failed( %struct.tree_node* %tmp507, i32 2, i8* getelementptr ([28 x i8]* @.str, i32 0, i64 0), i32 1755, i8* getelementptr ([12 x i8]* @__FUNCTION__.22683, i32 0, i32 0) )
	unreachable

cond_next527:		; preds = %cond_next500
	%tmp545 = and i32 %tmp513, 8192		; <i32> [#uses=1]
	%tmp547 = and i32 %tmp32, -8193		; <i32> [#uses=1]
	%tmp548 = or i32 %tmp547, %tmp545		; <i32> [#uses=1]
	store i32 %tmp548, i32* %tmp3031, align 8
	ret void

UnifiedReturnBlock:		; preds = %cond_next57, %cond_next46, %cond_false
	ret void
}

declare void @fancy_abort(i8*, i32, i8*)

declare void @tree_class_check_failed(%struct.tree_node*, i32, i8*, i32, i8*)

declare i32 @smallest_mode_for_size(i32, i32)

declare %struct.tree_node* @size_int_kind(i64, i32)

declare void @tree_check_failed(%struct.tree_node*, i8*, i32, i8*, ...)

declare i32 @tree_int_cst_sgn(%struct.tree_node*)
