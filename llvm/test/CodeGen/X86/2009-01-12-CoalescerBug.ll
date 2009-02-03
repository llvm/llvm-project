; RUN: llvm-as < %s | llc -mtriple=x86_64-unknown-linux-gnu | grep movq | count 2
; PR3311

	%struct.CUMULATIVE_ARGS = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.VEC_basic_block_base = type { i32, i32, [1 x %struct.basic_block_def*] }
	%struct.VEC_basic_block_gc = type { %struct.VEC_basic_block_base }
	%struct.VEC_edge_base = type { i32, i32, [1 x %struct.edge_def*] }
	%struct.VEC_edge_gc = type { %struct.VEC_edge_base }
	%struct.VEC_rtx_base = type { i32, i32, [1 x %struct.rtx_def*] }
	%struct.VEC_rtx_gc = type { %struct.VEC_rtx_base }
	%struct.VEC_temp_slot_p_base = type { i32, i32, [1 x %struct.temp_slot*] }
	%struct.VEC_temp_slot_p_gc = type { %struct.VEC_temp_slot_p_base }
	%struct.VEC_tree_base = type { i32, i32, [1 x %struct.tree_node*] }
	%struct.VEC_tree_gc = type { %struct.VEC_tree_base }
	%struct._obstack_chunk = type { i8*, %struct._obstack_chunk*, [4 x i8] }
	%struct.basic_block_def = type { %struct.tree_node*, %struct.VEC_edge_gc*, %struct.VEC_edge_gc*, i8*, %struct.loop*, [2 x %struct.et_node*], %struct.basic_block_def*, %struct.basic_block_def*, %struct.basic_block_il_dependent, %struct.tree_node*, %struct.edge_prediction*, i64, i32, i32, i32, i32 }
	%struct.basic_block_il_dependent = type { %struct.rtl_bb_info* }
	%struct.bitmap_element_def = type { %struct.bitmap_element_def*, %struct.bitmap_element_def*, i32, [2 x i64] }
	%struct.bitmap_head_def = type { %struct.bitmap_element_def*, %struct.bitmap_element_def*, i32, %struct.bitmap_obstack* }
	%struct.bitmap_obstack = type { %struct.bitmap_element_def*, %struct.bitmap_head_def*, %struct.obstack }
	%struct.block_symbol = type { [3 x %struct.rtunion], %struct.object_block*, i64 }
	%struct.c_arg_info = type { %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, i8 }
	%struct.c_language_function = type { %struct.stmt_tree_s }
	%struct.c_switch = type opaque
	%struct.control_flow_graph = type { %struct.basic_block_def*, %struct.basic_block_def*, %struct.VEC_basic_block_gc*, i32, i32, i32, %struct.VEC_basic_block_gc*, i32 }
	%struct.edge_def = type { %struct.basic_block_def*, %struct.basic_block_def*, %struct.edge_def_insns, i8*, %struct.location_t*, i32, i32, i64, i32 }
	%struct.edge_def_insns = type { %struct.rtx_def* }
	%struct.edge_prediction = type opaque
	%struct.eh_status = type opaque
	%struct.emit_status = type { i32, i32, %struct.rtx_def*, %struct.rtx_def*, %struct.sequence_stack*, i32, %struct.location_t, i32, i8*, %struct.rtx_def** }
	%struct.et_node = type opaque
	%struct.expr_status = type { i32, i32, i32, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def* }
	%struct.function = type { %struct.eh_status*, %struct.expr_status*, %struct.emit_status*, %struct.varasm_status*, %struct.control_flow_graph*, %struct.tree_node*, %struct.function*, i32, i32, i32, i32, %struct.rtx_def*, %struct.CUMULATIVE_ARGS, %struct.rtx_def*, %struct.rtx_def*, %struct.initial_value_struct*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, i8, i32, i64, %struct.tree_node*, %struct.tree_node*, %struct.rtx_def*, %struct.VEC_temp_slot_p_gc*, %struct.temp_slot*, %struct.var_refs_queue*, i32, i32, i32, i32, %struct.machine_function*, i32, i32, %struct.language_function*, %struct.htab*, %struct.rtx_def*, i32, i32, i32, %struct.location_t, %struct.VEC_tree_gc*, %struct.tree_node*, i8*, i8*, i8*, i8*, i8*, %struct.tree_node*, i8, i8, i8, i8, i8, i8 }
	%struct.htab = type { i32 (i8*)*, i32 (i8*, i8*)*, void (i8*)*, i8**, i64, i64, i64, i32, i32, i8* (i64, i64)*, void (i8*)*, i8*, i8* (i8*, i64, i64)*, void (i8*, i8*)*, i32 }
	%struct.initial_value_struct = type opaque
	%struct.lang_decl = type { i8 }
	%struct.language_function = type { %struct.c_language_function, %struct.tree_node*, %struct.tree_node*, %struct.c_switch*, %struct.c_arg_info*, i32, i32, i32, i32 }
	%struct.location_t = type { i8*, i32 }
	%struct.loop = type opaque
	%struct.machine_function = type { %struct.stack_local_entry*, i8*, %struct.rtx_def*, i32, i32, [4 x i32], i32, i32, i32 }
	%struct.object_block = type { %struct.section*, i32, i64, %struct.VEC_rtx_gc*, %struct.VEC_rtx_gc* }
	%struct.obstack = type { i64, %struct._obstack_chunk*, i8*, i8*, i8*, i64, i32, %struct._obstack_chunk* (i8*, i64)*, void (i8*, %struct._obstack_chunk*)*, i8*, i8 }
	%struct.omp_clause_subcode = type { i32 }
	%struct.rtl_bb_info = type { %struct.rtx_def*, %struct.rtx_def*, %struct.bitmap_head_def*, %struct.bitmap_head_def*, %struct.rtx_def*, %struct.rtx_def*, i32 }
	%struct.rtunion = type { i8* }
	%struct.rtx_def = type { i16, i8, i8, %struct.u }
	%struct.section = type { %struct.unnamed_section }
	%struct.sequence_stack = type { %struct.rtx_def*, %struct.rtx_def*, %struct.sequence_stack* }
	%struct.stack_local_entry = type opaque
	%struct.stmt_tree_s = type { %struct.tree_node*, i32 }
	%struct.temp_slot = type opaque
	%struct.tree_common = type { %struct.tree_node*, %struct.tree_node*, %union.tree_ann_d*, i8, i8, i8, i8, i8 }
	%struct.tree_decl_common = type { %struct.tree_decl_minimal, %struct.tree_node*, i8, i8, i8, i8, i8, i32, i32, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, i64, %struct.lang_decl* }
	%struct.tree_decl_minimal = type { %struct.tree_common, %struct.location_t, i32, %struct.tree_node*, %struct.tree_node* }
	%struct.tree_decl_non_common = type { %struct.tree_decl_with_vis, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node* }
	%struct.tree_decl_with_rtl = type { %struct.tree_decl_common, %struct.rtx_def*, i32 }
	%struct.tree_decl_with_vis = type { %struct.tree_decl_with_rtl, %struct.tree_node*, %struct.tree_node*, i8, i8, i8, i8 }
	%struct.tree_function_decl = type { %struct.tree_decl_non_common, i32, i8, i8, i64, %struct.function* }
	%struct.tree_node = type { %struct.tree_function_decl }
	%struct.u = type { %struct.block_symbol }
	%struct.unnamed_section = type { %struct.omp_clause_subcode, void (i8*)*, i8*, %struct.section* }
	%struct.var_refs_queue = type { %struct.rtx_def*, i32, i32, %struct.var_refs_queue* }
	%struct.varasm_status = type opaque
	%union.tree_ann_d = type opaque
@.str1 = external constant [31 x i8]		; <[31 x i8]*> [#uses=1]
@integer_types = external global [11 x %struct.tree_node*]		; <[11 x %struct.tree_node*]*> [#uses=1]
@__FUNCTION__.31164 = external constant [23 x i8], align 16		; <[23 x i8]*> [#uses=1]
@llvm.used = appending global [1 x i8*] [ i8* bitcast (i32 (i32, i32)* @c_common_type_for_size to i8*) ], section "llvm.metadata"		; <[1 x i8*]*> [#uses=0]

define i32 @c_common_type_for_size(i32 %bits, i32 %unsignedp) nounwind {
entry:
	%0 = load %struct.tree_node** getelementptr ([11 x %struct.tree_node*]* @integer_types, i32 0, i64 5), align 8		; <%struct.tree_node*> [#uses=1]
	br i1 false, label %bb16, label %bb

bb:		; preds = %entry
	tail call void @tree_class_check_failed(%struct.tree_node* %0, i32 2, i8* getelementptr ([31 x i8]* @.str1, i32 0, i64 0), i32 1785, i8* getelementptr ([23 x i8]* @__FUNCTION__.31164, i32 0, i32 0)) noreturn nounwind
	unreachable

bb16:		; preds = %entry
	%tmp = add i32 %bits, %unsignedp		; <i32> [#uses=1]
	ret i32 %tmp
}

declare void @tree_class_check_failed(%struct.tree_node*, i32, i8*, i32, i8*) noreturn
