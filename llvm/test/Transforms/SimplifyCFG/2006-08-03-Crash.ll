; RUN: opt < %s -passes=gvn,simplifycfg -simplifycfg-require-and-preserve-domtree=1 -disable-output
; PR867

target datalayout = "E-p:32:32"
target triple = "powerpc-unknown-linux-gnu"
	%struct.CUMULATIVE_ARGS = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.eh_status = type opaque
	%struct.emit_status = type { i32, i32, ptr, ptr, ptr, i32, %struct.location_t, i32, ptr, ptr }
	%struct.expr_status = type { i32, i32, i32, ptr, ptr, ptr }
	%struct.function = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, ptr, %struct.CUMULATIVE_ARGS, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i8, i32, i64, ptr, ptr, ptr, ptr, ptr, i32, ptr, i32, i32, ptr, ptr, i32, i32, i32, ptr, i32, i32, i8, i8, ptr, ptr, i32, i32, i32, i32, %struct.location_t, ptr, ptr, i8, i8, i8 }
	%struct.initial_value_struct = type opaque
	%struct.lang_decl = type opaque
	%struct.lang_type = type opaque
	%struct.language_function = type opaque
	%struct.location_t = type { ptr, i32 }
	%struct.machine_function = type { i32, i32, ptr, i32, i32 }
	%struct.rtunion = type { i32 }
	%struct.rtvec_def = type { i32, [1 x ptr] }
	%struct.rtx_def = type { i16, i8, i8, %struct.u }
	%struct.sequence_stack = type { ptr, ptr, ptr }
	%struct.temp_slot = type opaque
	%struct.tree_common = type { ptr, ptr, ptr, i8, i8, i8, i8, i8 }
	%struct.tree_decl = type { %struct.tree_common, %struct.location_t, i32, ptr, i8, i8, i8, i8, i8, i8, i8, i8, i32, %struct.tree_decl_u1, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, %struct.tree_decl_u2, ptr, ptr, i64, ptr }
	%struct.tree_decl_u1 = type { i64 }
	%struct.tree_decl_u2 = type { ptr }
	%struct.tree_node = type { %struct.tree_decl }
	%struct.tree_type = type { %struct.tree_common, ptr, ptr, ptr, ptr, i32, i16, i8, i8, i32, ptr, ptr, %struct.rtunion, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i64, ptr }
	%struct.u = type { [1 x i64] }
	%struct.var_refs_queue = type { ptr, i32, i32, ptr }
	%struct.varasm_status = type opaque
	%struct.varray_head_tag = type { i32, i32, i32, ptr, %struct.u }
	%union.tree_ann_d = type opaque
@mode_class = external global [35 x i8]		; <ptr> [#uses=3]

define void @fold_builtin_classify() {
entry:
	%tmp63 = load i32, ptr null		; <i32> [#uses=1]
	switch i32 %tmp63, label %bb276 [
		 i32 414, label %bb145
		 i32 417, label %bb
	]
bb:		; preds = %entry
	ret void
bb145:		; preds = %entry
	%tmp146 = load ptr, ptr null		; <ptr> [#uses=1]
	%tmp148 = getelementptr %struct.tree_node, ptr %tmp146, i32 0, i32 0, i32 0, i32 1		; <ptr> [#uses=1]
	%tmp149 = load ptr, ptr %tmp148		; <ptr> [#uses=1]
	%tmp151 = getelementptr %struct.tree_type, ptr %tmp149, i32 0, i32 6		; <ptr> [#uses=1]
	%tmp152 = load i32, ptr %tmp151		; <i32> [#uses=1]
	%tmp154 = lshr i32 %tmp152, 16		; <i32> [#uses=1]
	%tmp154.mask = and i32 %tmp154, 127		; <i32> [#uses=1]
	%gep.upgrd.2 = zext i32 %tmp154.mask to i64		; <i64> [#uses=1]
	%tmp155 = getelementptr [35 x i8], ptr @mode_class, i32 0, i64 %gep.upgrd.2		; <ptr> [#uses=1]
	%tmp156 = load i8, ptr %tmp155		; <i8> [#uses=1]
	%tmp157 = icmp eq i8 %tmp156, 4		; <i1> [#uses=1]
	br i1 %tmp157, label %cond_next241, label %cond_true158
cond_true158:		; preds = %bb145
	%tmp172 = load ptr, ptr null		; <ptr> [#uses=1]
	%tmp174 = getelementptr %struct.tree_node, ptr %tmp172, i32 0, i32 0, i32 0, i32 1		; <ptr> [#uses=1]
	%tmp175 = load ptr, ptr %tmp174		; <ptr> [#uses=1]
	%tmp177 = getelementptr %struct.tree_type, ptr %tmp175, i32 0, i32 6		; <ptr> [#uses=1]
	%tmp178 = load i32, ptr %tmp177		; <i32> [#uses=1]
	%tmp180 = lshr i32 %tmp178, 16		; <i32> [#uses=1]
	%tmp180.mask = and i32 %tmp180, 127		; <i32> [#uses=1]
	%gep.upgrd.4 = zext i32 %tmp180.mask to i64		; <i64> [#uses=1]
	%tmp181 = getelementptr [35 x i8], ptr @mode_class, i32 0, i64 %gep.upgrd.4		; <ptr> [#uses=1]
	%tmp182 = load i8, ptr %tmp181		; <i8> [#uses=1]
	%tmp183 = icmp eq i8 %tmp182, 8		; <i1> [#uses=1]
	br i1 %tmp183, label %cond_next241, label %cond_true184
cond_true184:		; preds = %cond_true158
	%tmp185 = load ptr, ptr null		; <ptr> [#uses=1]
	%tmp187 = getelementptr %struct.tree_node, ptr %tmp185, i32 0, i32 0, i32 0, i32 1		; <ptr> [#uses=1]
	%tmp188 = load ptr, ptr %tmp187		; <ptr> [#uses=1]
	%tmp190 = getelementptr %struct.tree_type, ptr %tmp188, i32 0, i32 6		; <ptr> [#uses=1]
	%tmp191 = load i32, ptr %tmp190		; <i32> [#uses=1]
	%tmp193 = lshr i32 %tmp191, 16		; <i32> [#uses=1]
	%tmp193.mask = and i32 %tmp193, 127		; <i32> [#uses=1]
	%gep.upgrd.6 = zext i32 %tmp193.mask to i64		; <i64> [#uses=1]
	%tmp194 = getelementptr [35 x i8], ptr @mode_class, i32 0, i64 %gep.upgrd.6		; <ptr> [#uses=1]
	%tmp195 = load i8, ptr %tmp194		; <i8> [#uses=1]
	%tmp196 = icmp eq i8 %tmp195, 4		; <i1> [#uses=1]
	br i1 %tmp196, label %cond_next241, label %cond_true197
cond_true197:		; preds = %cond_true184
	ret void
cond_next241:		; preds = %cond_true184, %cond_true158, %bb145
	%tmp245 = load i32, ptr null		; <i32> [#uses=0]
	ret void
bb276:		; preds = %entry
	ret void
}
