; RUN: llc < %s -mtriple=x86_64-linux-gnu
; PR1729

	%struct.CUMULATIVE_ARGS = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.VEC_edge = type { i32, i32, [1 x ptr] }
	%struct.VEC_tree = type { i32, i32, [1 x ptr] }
	%struct._IO_FILE = type { i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, [1 x i8], ptr, i64, ptr, ptr, ptr, ptr, i64, i32, [20 x i8] }
	%struct._IO_marker = type { ptr, ptr, i32 }
	%struct._obstack_chunk = type { ptr, ptr, [4 x i8] }
	%struct.addr_diff_vec_flags = type <{ i8, i8, i8, i8 }>
	%struct.alloc_pool_def = type { ptr, i64, i64, ptr, i64, i64, i64, ptr, i64, i64 }
	%struct.alloc_pool_list_def = type { ptr }
	%struct.basic_block_def = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, [2 x ptr], ptr, ptr, ptr, ptr, i64, i32, i32, i32, i32 }
	%struct.bb_ann_d = type opaque
	%struct.bitmap_element_def = type { ptr, ptr, i32, [2 x i64] }
	%struct.bitmap_head_def = type { ptr, ptr, i32, ptr }
	%struct.bitmap_obstack = type { ptr, ptr, %struct.obstack }
	%struct.cselib_val_struct = type opaque
	%struct.dataflow_d = type opaque
	%struct.die_struct = type opaque
	%struct.edge_def = type { ptr, ptr, %struct.edge_def_insns, ptr, ptr, i32, i32, i64, i32 }
	%struct.edge_def_insns = type { ptr }
	%struct.edge_iterator = type { i32, ptr }
	%struct.eh_status = type opaque
	%struct.elt_list = type opaque
	%struct.emit_status = type { i32, i32, ptr, ptr, ptr, i32, %struct.location_t, i32, ptr, ptr }
	%struct.et_node = type opaque
	%struct.expr_status = type { i32, i32, i32, ptr, ptr, ptr }
	%struct.function = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, ptr, %struct.CUMULATIVE_ARGS, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i8, i32, i64, ptr, ptr, ptr, ptr, ptr, i32, ptr, i32, i32, ptr, ptr, i32, i32, i32, ptr, i32, i32, i8, i8, ptr, ptr, i32, i32, i32, i32, %struct.location_t, ptr, ptr, ptr, i8, i8, i8 }
	%struct.ht_identifier = type { ptr, i32, i32 }
	%struct.initial_value_struct = type opaque
	%struct.lang_decl = type opaque
	%struct.lang_type = type opaque
	%struct.language_function = type opaque
	%struct.location_t = type { ptr, i32 }
	%struct.loop = type opaque
	%struct.machine_function = type { ptr, ptr, ptr, i32, i32, i32, i32, i32 }
	%struct.mem_attrs = type { i64, ptr, ptr, ptr, i32 }
	%struct.obstack = type { i64, ptr, ptr, ptr, ptr, i64, i32, ptr, ptr, ptr, i8 }
	%struct.phi_arg_d = type { ptr, i8 }
	%struct.ptr_info_def = type opaque
	%struct.real_value = type opaque
	%struct.reg_attrs = type { ptr, i64 }
	%struct.reg_info_def = type { i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.reorder_block_def = type { ptr, ptr, ptr, ptr, ptr, i32, i32, i32 }
	%struct.rtunion = type { ptr }
	%struct.rtvec_def = type { i32, [1 x ptr] }
	%struct.rtx_def = type { i16, i8, i8, %struct.u }
	%struct.sequence_stack = type { ptr, ptr, ptr }
	%struct.simple_bitmap_def = type { i32, i32, i32, [1 x i64] }
	%struct.stack_local_entry = type opaque
	%struct.temp_slot = type opaque
	%struct.tree_binfo = type { %struct.tree_common, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, %struct.VEC_tree }
	%struct.tree_block = type { %struct.tree_common, i32, ptr, ptr, ptr, ptr, ptr, ptr }
	%struct.tree_common = type { ptr, ptr, ptr, i8, i8, i8, i8, i8 }
	%struct.tree_complex = type { %struct.tree_common, ptr, ptr }
	%struct.tree_decl = type { %struct.tree_common, %struct.location_t, i32, ptr, i8, i8, i8, i8, i8, i8, i8, i8, i32, %struct.tree_decl_u1, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, %struct.tree_decl_u2, ptr, ptr, i64, ptr }
	%struct.tree_decl_u1 = type { i64 }
	%struct.tree_decl_u1_a = type <{ i32 }>
	%struct.tree_decl_u2 = type { ptr }
	%struct.tree_exp = type { %struct.tree_common, ptr, i32, ptr, [1 x ptr] }
	%struct.tree_identifier = type { %struct.tree_common, %struct.ht_identifier }
	%struct.tree_int_cst = type { %struct.tree_common, %struct.tree_int_cst_lowhi }
	%struct.tree_int_cst_lowhi = type { i64, i64 }
	%struct.tree_list = type { %struct.tree_common, ptr, ptr }
	%struct.tree_node = type { %struct.tree_decl }
	%struct.tree_phi_node = type { %struct.tree_common, ptr, i32, i32, i32, ptr, ptr, [1 x %struct.phi_arg_d] }
	%struct.tree_real_cst = type { %struct.tree_common, ptr }
	%struct.tree_ssa_name = type { %struct.tree_common, ptr, i32, ptr, ptr, ptr }
	%struct.tree_statement_list = type { %struct.tree_common, ptr, ptr }
	%struct.tree_statement_list_node = type { ptr, ptr, ptr }
	%struct.tree_string = type { %struct.tree_common, i32, [1 x i8] }
	%struct.tree_type = type { %struct.tree_common, ptr, ptr, ptr, ptr, i32, i16, i8, i8, i32, ptr, ptr, %struct.rtunion, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i64, ptr }
	%struct.tree_type_symtab = type { ptr }
	%struct.tree_value_handle = type { %struct.tree_common, ptr, i32 }
	%struct.tree_vec = type { %struct.tree_common, i32, [1 x ptr] }
	%struct.tree_vector = type { %struct.tree_common, ptr }
	%struct.u = type { [1 x %struct.rtunion] }
	%struct.value_set = type opaque
	%struct.var_refs_queue = type { ptr, i32, i32, ptr }
	%struct.varasm_status = type opaque
	%struct.varray_data = type { [1 x i64] }
	%struct.varray_head_tag = type { i64, i64, i32, ptr, %struct.varray_data }
	%union.tree_ann_d = type opaque
@first_edge_aux_obj = external global ptr		; <ptr> [#uses=0]
@first_block_aux_obj = external global ptr		; <ptr> [#uses=0]
@n_edges = external global i32		; <ptr> [#uses=0]
@ENTRY_BLOCK_PTR = external global ptr		; <ptr> [#uses=0]
@EXIT_BLOCK_PTR = external global ptr		; <ptr> [#uses=0]
@n_basic_blocks = external global i32		; <ptr> [#uses=0]
@.str = external constant [9 x i8]		; <ptr> [#uses=0]
@rbi_pool = external global ptr		; <ptr> [#uses=0]
@__FUNCTION__.19643 = external constant [18 x i8]		; <ptr> [#uses=0]
@.str1 = external constant [20 x i8]		; <ptr> [#uses=0]
@__FUNCTION__.19670 = external constant [15 x i8]		; <ptr> [#uses=0]
@basic_block_info = external global ptr		; <ptr> [#uses=0]
@last_basic_block = external global i32		; <ptr> [#uses=0]
@__FUNCTION__.19696 = external constant [14 x i8]		; <ptr> [#uses=0]
@__FUNCTION__.20191 = external constant [20 x i8]		; <ptr> [#uses=0]
@block_aux_obstack = external global %struct.obstack		; <ptr> [#uses=0]
@__FUNCTION__.20301 = external constant [20 x i8]		; <ptr> [#uses=0]
@__FUNCTION__.20316 = external constant [19 x i8]		; <ptr> [#uses=0]
@edge_aux_obstack = external global %struct.obstack		; <ptr> [#uses=0]
@stderr = external global ptr		; <ptr> [#uses=0]
@__FUNCTION__.20463 = external constant [11 x i8]		; <ptr> [#uses=0]
@.str2 = external constant [7 x i8]		; <ptr> [#uses=0]
@.str3 = external constant [6 x i8]		; <ptr> [#uses=0]
@.str4 = external constant [4 x i8]		; <ptr> [#uses=0]
@.str5 = external constant [11 x i8]		; <ptr> [#uses=0]
@.str6 = external constant [8 x i8]		; <ptr> [#uses=0]
@.str7 = external constant [4 x i8]		; <ptr> [#uses=0]
@bitnames.20157 = external constant [13 x ptr]		; <ptr> [#uses=0]
@.str8 = external constant [9 x i8]		; <ptr> [#uses=0]
@.str9 = external constant [3 x i8]		; <ptr> [#uses=0]
@.str10 = external constant [7 x i8]		; <ptr> [#uses=0]
@.str11 = external constant [3 x i8]		; <ptr> [#uses=0]
@.str12 = external constant [5 x i8]		; <ptr> [#uses=0]
@.str13 = external constant [9 x i8]		; <ptr> [#uses=0]
@.str14 = external constant [13 x i8]		; <ptr> [#uses=0]
@.str15 = external constant [12 x i8]		; <ptr> [#uses=0]
@.str16 = external constant [8 x i8]		; <ptr> [#uses=0]
@.str17 = external constant [10 x i8]		; <ptr> [#uses=0]
@.str18 = external constant [5 x i8]		; <ptr> [#uses=0]
@.str19 = external constant [6 x i8]		; <ptr> [#uses=0]
@.str20 = external constant [5 x i8]		; <ptr> [#uses=0]
@.str21 = external constant [3 x i8]		; <ptr> [#uses=0]
@.str22 = external constant [3 x i8]		; <ptr> [#uses=0]
@__FUNCTION__.19709 = external constant [20 x i8]		; <ptr> [#uses=0]
@.str23 = external constant [5 x i8]		; <ptr> [#uses=0]
@.str24 = external constant [10 x i8]		; <ptr> [#uses=0]
@__FUNCTION__.19813 = external constant [19 x i8]		; <ptr> [#uses=0]
@.str25 = external constant [7 x i8]		; <ptr> [#uses=0]
@.str26 = external constant [6 x i8]		; <ptr> [#uses=0]
@initialized.20241.b = external global i1		; <ptr> [#uses=0]
@__FUNCTION__.20244 = external constant [21 x i8]		; <ptr> [#uses=0]
@__FUNCTION__.19601 = external constant [12 x i8]		; <ptr> [#uses=0]
@__FUNCTION__.14571 = external constant [8 x i8]		; <ptr> [#uses=0]
@__FUNCTION__.14535 = external constant [13 x i8]		; <ptr> [#uses=0]
@.str27 = external constant [28 x i8]		; <ptr> [#uses=0]
@__FUNCTION__.14589 = external constant [8 x i8]		; <ptr> [#uses=0]
@__FUNCTION__.19792 = external constant [12 x i8]		; <ptr> [#uses=0]
@__FUNCTION__.19851 = external constant [19 x i8]		; <ptr> [#uses=0]
@profile_status = external global i32		; <ptr> [#uses=0]
@.str29 = external constant [46 x i8]		; <ptr> [#uses=0]
@.str30 = external constant [49 x i8]		; <ptr> [#uses=0]
@.str31 = external constant [54 x i8]		; <ptr> [#uses=0]
@.str32 = external constant [49 x i8]		; <ptr> [#uses=1]
@__FUNCTION__.19948 = external constant [15 x i8]		; <ptr> [#uses=0]
@reg_n_info = external global ptr		; <ptr> [#uses=0]
@reload_completed = external global i32		; <ptr> [#uses=0]
@.str33 = external constant [15 x i8]		; <ptr> [#uses=0]
@.str34 = external constant [43 x i8]		; <ptr> [#uses=0]
@.str35 = external constant [13 x i8]		; <ptr> [#uses=0]
@.str36 = external constant [1 x i8]		; <ptr> [#uses=0]
@.str37 = external constant [2 x i8]		; <ptr> [#uses=0]
@.str38 = external constant [16 x i8]		; <ptr> [#uses=0]
@cfun = external global ptr		; <ptr> [#uses=0]
@.str39 = external constant [14 x i8]		; <ptr> [#uses=0]
@.str40 = external constant [11 x i8]		; <ptr> [#uses=0]
@.str41 = external constant [20 x i8]		; <ptr> [#uses=0]
@.str42 = external constant [17 x i8]		; <ptr> [#uses=0]
@.str43 = external constant [19 x i8]		; <ptr> [#uses=0]
@mode_size = external global [48 x i8]		; <ptr> [#uses=0]
@target_flags = external global i32		; <ptr> [#uses=0]
@.str44 = external constant [11 x i8]		; <ptr> [#uses=0]
@reg_class_names = external global [0 x ptr]		; <ptr> [#uses=0]
@.str45 = external constant [10 x i8]		; <ptr> [#uses=0]
@.str46 = external constant [13 x i8]		; <ptr> [#uses=0]
@.str47 = external constant [19 x i8]		; <ptr> [#uses=0]
@.str48 = external constant [12 x i8]		; <ptr> [#uses=0]
@.str49 = external constant [10 x i8]		; <ptr> [#uses=0]
@.str50 = external constant [3 x i8]		; <ptr> [#uses=0]
@.str51 = external constant [29 x i8]		; <ptr> [#uses=0]
@.str52 = external constant [17 x i8]		; <ptr> [#uses=0]
@.str53 = external constant [19 x i8]		; <ptr> [#uses=0]
@.str54 = external constant [22 x i8]		; <ptr> [#uses=0]
@.str55 = external constant [10 x i8]		; <ptr> [#uses=0]
@.str56 = external constant [12 x i8]		; <ptr> [#uses=0]
@.str57 = external constant [26 x i8]		; <ptr> [#uses=0]
@.str58 = external constant [15 x i8]		; <ptr> [#uses=0]
@.str59 = external constant [14 x i8]		; <ptr> [#uses=0]
@.str60 = external constant [26 x i8]		; <ptr> [#uses=0]
@.str61 = external constant [24 x i8]		; <ptr> [#uses=0]
@initialized.20366.b = external global i1		; <ptr> [#uses=0]
@__FUNCTION__.20369 = external constant [20 x i8]		; <ptr> [#uses=0]
@__FUNCTION__.20442 = external constant [19 x i8]		; <ptr> [#uses=0]
@bb_bitnames.20476 = external constant [6 x ptr]		; <ptr> [#uses=0]
@.str62 = external constant [6 x i8]		; <ptr> [#uses=0]
@.str63 = external constant [4 x i8]		; <ptr> [#uses=0]
@.str64 = external constant [10 x i8]		; <ptr> [#uses=0]
@.str65 = external constant [8 x i8]		; <ptr> [#uses=0]
@.str66 = external constant [17 x i8]		; <ptr> [#uses=0]
@.str67 = external constant [11 x i8]		; <ptr> [#uses=0]
@.str68 = external constant [15 x i8]		; <ptr> [#uses=0]
@.str69 = external constant [3 x i8]		; <ptr> [#uses=0]
@.str70 = external constant [3 x i8]		; <ptr> [#uses=0]
@__FUNCTION__.20520 = external constant [32 x i8]		; <ptr> [#uses=0]
@dump_file = external global ptr		; <ptr> [#uses=0]
@.str71 = external constant [86 x i8]		; <ptr> [#uses=0]
@.str72 = external constant [94 x i8]		; <ptr> [#uses=0]
@reg_obstack = external global %struct.bitmap_obstack		; <ptr> [#uses=0]

declare void @init_flow()

declare ptr @ggc_alloc_cleared_stat(i64)

declare fastcc void @free_edge(ptr)

declare void @ggc_free(ptr)

declare ptr @alloc_block()

declare void @alloc_rbi_pool()

declare ptr @create_alloc_pool(ptr, i64, i64)

declare void @free_rbi_pool()

declare void @free_alloc_pool(ptr)

declare void @initialize_bb_rbi(ptr)

declare void @fancy_abort(ptr, i32, ptr)

declare ptr @pool_alloc(ptr)

declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1)

declare void @link_block(ptr, ptr)

declare void @unlink_block(ptr)

declare void @compact_blocks()

declare void @varray_check_failed(ptr, i64, ptr, i32, ptr)

declare void @expunge_block(ptr)

declare void @clear_bb_flags()

declare void @alloc_aux_for_block(ptr, i32)

declare void @_obstack_newchunk(ptr, i32)

declare void @clear_aux_for_blocks()

declare void @free_aux_for_blocks()

declare void @obstack_free(ptr, ptr)

declare void @alloc_aux_for_edge(ptr, i32)

declare void @debug_bb(ptr)

declare void @dump_bb(ptr, ptr, i32)

declare ptr @debug_bb_n(i32)

declare void @dump_edge_info(ptr, ptr, i32)

declare i32 @fputs_unlocked(ptr noalias , ptr noalias )

declare i32 @fprintf(ptr noalias , ptr noalias , ...)

declare i64 @fwrite(ptr, i64, i64, ptr)

declare i32 @__overflow(ptr, i32)

declare ptr @unchecked_make_edge(ptr, ptr, i32)

declare ptr @vec_gc_p_reserve(ptr, i32)

declare void @vec_assert_fail(ptr, ptr, ptr, i32, ptr)

declare void @execute_on_growing_pred(ptr)

declare ptr @make_edge(ptr, ptr, i32)

declare ptr @find_edge(ptr, ptr)

declare ptr @make_single_succ_edge(ptr, ptr, i32)

declare ptr @cached_make_edge(ptr, ptr, ptr, i32)

declare void @redirect_edge_succ(ptr, ptr)

declare void @execute_on_shrinking_pred(ptr)

declare void @alloc_aux_for_blocks(i32)

declare ptr @xmalloc(i64)

declare i32 @_obstack_begin(ptr, i32, i32, ptr, ptr)

declare void @free(ptr)

declare void @clear_edges()

declare void @remove_edge(ptr)

declare ptr @redirect_edge_succ_nodup(ptr, ptr)

declare void @redirect_edge_pred(ptr, ptr)

define void @check_bb_profile(ptr %bb, ptr %file) {
entry:
	br i1 false, label %cond_false759.preheader, label %cond_false149.preheader

cond_false149.preheader:		; preds = %entry
	ret void

cond_false759.preheader:		; preds = %entry
	br i1 false, label %cond_next873, label %cond_true794

bb644:		; preds = %cond_next873
	ret void

cond_true794:		; preds = %cond_false759.preheader
	ret void

cond_next873:		; preds = %cond_false759.preheader
	br i1 false, label %bb882, label %bb644

bb882:		; preds = %cond_next873
	br i1 false, label %cond_true893, label %cond_next901

cond_true893:		; preds = %bb882
	br label %cond_false1036

cond_next901:		; preds = %bb882
	ret void

bb929:		; preds = %cond_next1150
	%tmp934 = add i64 0, %lsum.11225.0		; <i64> [#uses=1]
	br i1 false, label %cond_next979, label %cond_true974

cond_true974:		; preds = %bb929
	ret void

cond_next979:		; preds = %bb929
	br label %cond_false1036

cond_false1036:		; preds = %cond_next979, %cond_true893
	%lsum.11225.0 = phi i64 [ 0, %cond_true893 ], [ %tmp934, %cond_next979 ]		; <i64> [#uses=2]
	br i1 false, label %cond_next1056, label %cond_true1051

cond_true1051:		; preds = %cond_false1036
	ret void

cond_next1056:		; preds = %cond_false1036
	br i1 false, label %cond_next1150, label %cond_true1071

cond_true1071:		; preds = %cond_next1056
	ret void

cond_next1150:		; preds = %cond_next1056
	%tmp1156 = icmp eq ptr null, null		; <i1> [#uses=1]
	br i1 %tmp1156, label %bb1159, label %bb929

bb1159:		; preds = %cond_next1150
	br i1 false, label %cond_true1169, label %UnifiedReturnBlock

cond_true1169:		; preds = %bb1159
	%tmp11741175 = trunc i64 %lsum.11225.0 to i32		; <i32> [#uses=1]
	%tmp1178 = tail call i32 (ptr  , ptr  , ...) @fprintf( ptr noalias %file  , ptr @.str32  , i32 %tmp11741175, i32 0 )		; <i32> [#uses=0]
	ret void

UnifiedReturnBlock:		; preds = %bb1159
	ret void
}

declare void @dump_flow_info(ptr)

declare i32 @max_reg_num()

declare void @rtl_check_failed_flag(ptr, ptr, ptr, i32, ptr)

declare i32 @reg_preferred_class(i32)

declare i32 @reg_alternate_class(i32)

declare zeroext i8 @maybe_hot_bb_p(ptr)  

declare zeroext i8 @probably_never_executed_bb_p(ptr)  

declare void @dump_regset(ptr, ptr)

declare void @debug_flow_info()

declare void @alloc_aux_for_edges(i32)

declare void @clear_aux_for_edges()

declare void @free_aux_for_edges()

declare void @brief_dump_cfg(ptr)

declare i32 @fputc(i32, ptr)

declare void @update_bb_profile_for_threading(ptr, i32, i64, ptr)
