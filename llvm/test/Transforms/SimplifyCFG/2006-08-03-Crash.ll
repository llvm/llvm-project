; RUN: opt < %s -gvn -simplifycfg -disable-output
; PR867

target datalayout = "E-p:32:32"
target triple = "powerpc-apple-darwin8"
	%struct.CUMULATIVE_ARGS = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.eh_status = type opaque
	%struct.emit_status = type { i32, i32, %struct.rtx_def*, %struct.rtx_def*, %struct.sequence_stack*, i32, %struct.location_t, i32, i8*, %struct.rtx_def** }
	%struct.expr_status = type { i32, i32, i32, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def* }
	%struct.function = type { %struct.eh_status*, %struct.expr_status*, %struct.emit_status*, %struct.varasm_status*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.function*, i32, i32, i32, i32, %struct.rtx_def*, %struct.CUMULATIVE_ARGS, %struct.rtx_def*, %struct.rtx_def*, %struct.initial_value_struct*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, %struct.rtx_def*, i8, i32, i64, %struct.tree_node*, %struct.tree_node*, %struct.rtx_def*, %struct.varray_head_tag*, %struct.temp_slot*, i32, %struct.var_refs_queue*, i32, i32, %struct.rtvec_def*, %struct.tree_node*, i32, i32, i32, %struct.machine_function*, i32, i32, i8, i8, %struct.language_function*, %struct.rtx_def*, i32, i32, i32, i32, %struct.location_t, %struct.varray_head_tag*, %struct.tree_node*, i8, i8, i8 }
	%struct.initial_value_struct = type opaque
	%struct.lang_decl = type opaque
	%struct.lang_type = type opaque
	%struct.language_function = type opaque
	%struct.location_t = type { i8*, i32 }
	%struct.machine_function = type { i32, i32, i8*, i32, i32 }
	%struct.rtunion = type { i32 }
	%struct.rtvec_def = type { i32, [1 x %struct.rtx_def*] }
	%struct.rtx_def = type { i16, i8, i8, %struct.u }
	%struct.sequence_stack = type { %struct.rtx_def*, %struct.rtx_def*, %struct.sequence_stack* }
	%struct.temp_slot = type opaque
	%struct.tree_common = type { %struct.tree_node*, %struct.tree_node*, %union.tree_ann_d*, i8, i8, i8, i8, i8 }
	%struct.tree_decl = type { %struct.tree_common, %struct.location_t, i32, %struct.tree_node*, i8, i8, i8, i8, i8, i8, i8, i8, i32, %struct.tree_decl_u1, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.rtx_def*, %struct.tree_decl_u2, %struct.tree_node*, %struct.tree_node*, i64, %struct.lang_decl* }
	%struct.tree_decl_u1 = type { i64 }
	%struct.tree_decl_u2 = type { %struct.function* }
	%struct.tree_node = type { %struct.tree_decl }
	%struct.tree_type = type { %struct.tree_common, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, i32, i16, i8, i8, i32, %struct.tree_node*, %struct.tree_node*, %struct.rtunion, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, i64, %struct.lang_type* }
	%struct.u = type { [1 x i64] }
	%struct.var_refs_queue = type { %struct.rtx_def*, i32, i32, %struct.var_refs_queue* }
	%struct.varasm_status = type opaque
	%struct.varray_head_tag = type { i32, i32, i32, i8*, %struct.u }
	%union.tree_ann_d = type opaque
@mode_class = external global [35 x i8]		; <[35 x i8]*> [#uses=3]

define void @fold_builtin_classify() {
entry:
	%tmp63 = load i32* null		; <i32> [#uses=1]
	switch i32 %tmp63, label %bb276 [
		 i32 414, label %bb145
		 i32 417, label %bb
	]
bb:		; preds = %entry
	ret void
bb145:		; preds = %entry
	%tmp146 = load %struct.tree_node** null		; <%struct.tree_node*> [#uses=1]
	%tmp148 = getelementptr %struct.tree_node* %tmp146, i32 0, i32 0, i32 0, i32 1		; <%struct.tree_node**> [#uses=1]
	%tmp149 = load %struct.tree_node** %tmp148		; <%struct.tree_node*> [#uses=1]
	%tmp150 = bitcast %struct.tree_node* %tmp149 to %struct.tree_type*		; <%struct.tree_type*> [#uses=1]
	%tmp151 = getelementptr %struct.tree_type* %tmp150, i32 0, i32 6		; <i16*> [#uses=1]
	%tmp151.upgrd.1 = bitcast i16* %tmp151 to i32*		; <i32*> [#uses=1]
	%tmp152 = load i32* %tmp151.upgrd.1		; <i32> [#uses=1]
	%tmp154 = lshr i32 %tmp152, 16		; <i32> [#uses=1]
	%tmp154.mask = and i32 %tmp154, 127		; <i32> [#uses=1]
	%gep.upgrd.2 = zext i32 %tmp154.mask to i64		; <i64> [#uses=1]
	%tmp155 = getelementptr [35 x i8]* @mode_class, i32 0, i64 %gep.upgrd.2		; <i8*> [#uses=1]
	%tmp156 = load i8* %tmp155		; <i8> [#uses=1]
	%tmp157 = icmp eq i8 %tmp156, 4		; <i1> [#uses=1]
	br i1 %tmp157, label %cond_next241, label %cond_true158
cond_true158:		; preds = %bb145
	%tmp172 = load %struct.tree_node** null		; <%struct.tree_node*> [#uses=1]
	%tmp174 = getelementptr %struct.tree_node* %tmp172, i32 0, i32 0, i32 0, i32 1		; <%struct.tree_node**> [#uses=1]
	%tmp175 = load %struct.tree_node** %tmp174		; <%struct.tree_node*> [#uses=1]
	%tmp176 = bitcast %struct.tree_node* %tmp175 to %struct.tree_type*		; <%struct.tree_type*> [#uses=1]
	%tmp177 = getelementptr %struct.tree_type* %tmp176, i32 0, i32 6		; <i16*> [#uses=1]
	%tmp177.upgrd.3 = bitcast i16* %tmp177 to i32*		; <i32*> [#uses=1]
	%tmp178 = load i32* %tmp177.upgrd.3		; <i32> [#uses=1]
	%tmp180 = lshr i32 %tmp178, 16		; <i32> [#uses=1]
	%tmp180.mask = and i32 %tmp180, 127		; <i32> [#uses=1]
	%gep.upgrd.4 = zext i32 %tmp180.mask to i64		; <i64> [#uses=1]
	%tmp181 = getelementptr [35 x i8]* @mode_class, i32 0, i64 %gep.upgrd.4		; <i8*> [#uses=1]
	%tmp182 = load i8* %tmp181		; <i8> [#uses=1]
	%tmp183 = icmp eq i8 %tmp182, 8		; <i1> [#uses=1]
	br i1 %tmp183, label %cond_next241, label %cond_true184
cond_true184:		; preds = %cond_true158
	%tmp185 = load %struct.tree_node** null		; <%struct.tree_node*> [#uses=1]
	%tmp187 = getelementptr %struct.tree_node* %tmp185, i32 0, i32 0, i32 0, i32 1		; <%struct.tree_node**> [#uses=1]
	%tmp188 = load %struct.tree_node** %tmp187		; <%struct.tree_node*> [#uses=1]
	%tmp189 = bitcast %struct.tree_node* %tmp188 to %struct.tree_type*		; <%struct.tree_type*> [#uses=1]
	%tmp190 = getelementptr %struct.tree_type* %tmp189, i32 0, i32 6		; <i16*> [#uses=1]
	%tmp190.upgrd.5 = bitcast i16* %tmp190 to i32*		; <i32*> [#uses=1]
	%tmp191 = load i32* %tmp190.upgrd.5		; <i32> [#uses=1]
	%tmp193 = lshr i32 %tmp191, 16		; <i32> [#uses=1]
	%tmp193.mask = and i32 %tmp193, 127		; <i32> [#uses=1]
	%gep.upgrd.6 = zext i32 %tmp193.mask to i64		; <i64> [#uses=1]
	%tmp194 = getelementptr [35 x i8]* @mode_class, i32 0, i64 %gep.upgrd.6		; <i8*> [#uses=1]
	%tmp195 = load i8* %tmp194		; <i8> [#uses=1]
	%tmp196 = icmp eq i8 %tmp195, 4		; <i1> [#uses=1]
	br i1 %tmp196, label %cond_next241, label %cond_true197
cond_true197:		; preds = %cond_true184
	ret void
cond_next241:		; preds = %cond_true184, %cond_true158, %bb145
	%tmp245 = load i32* null		; <i32> [#uses=0]
	ret void
bb276:		; preds = %entry
	ret void
}
