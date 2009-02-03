; RUN: llvm-as < %s | llc -march=x86
	%struct.function = type opaque
	%struct.lang_decl = type opaque
	%struct.location_t = type { i8*, i32 }
	%struct.rtx_def = type opaque
	%struct.tree_common = type { %struct.tree_node*, %struct.tree_node*, %union.tree_ann_d*, i8, i8, i8, i8, i8 }
	%struct.tree_decl = type { %struct.tree_common, %struct.location_t, i32, %struct.tree_node*, i8, i8, i8, i8, i8, i8, i8, i8, i32, %struct.tree_decl_u1, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.tree_node*, %struct.rtx_def*, i32, %struct.tree_decl_u2, %struct.tree_node*, %struct.tree_node*, i64, %struct.lang_decl* }
	%struct.tree_decl_u1 = type { i64 }
	%struct.tree_decl_u2 = type { %struct.function* }
	%struct.tree_node = type { %struct.tree_decl }
	%union.tree_ann_d = type opaque

define void @check_format_arg() {
	br i1 false, label %cond_next196, label %bb12.preheader

bb12.preheader:		; preds = %0
	ret void

cond_next196:		; preds = %0
	br i1 false, label %cond_next330, label %cond_true304

cond_true304:		; preds = %cond_next196
	ret void

cond_next330:		; preds = %cond_next196
	br i1 false, label %cond_next472, label %bb441

bb441:		; preds = %cond_next330
	ret void

cond_next472:		; preds = %cond_next330
	%tmp490 = load %struct.tree_node** null		; <%struct.tree_node*> [#uses=1]
	%tmp492 = getelementptr %struct.tree_node* %tmp490, i32 0, i32 0, i32 0, i32 3		; <i8*> [#uses=1]
	%tmp492.upgrd.1 = bitcast i8* %tmp492 to i32*		; <i32*> [#uses=1]
	%tmp493 = load i32* %tmp492.upgrd.1		; <i32> [#uses=1]
	%tmp495 = trunc i32 %tmp493 to i8		; <i8> [#uses=1]
	%tmp496 = icmp eq i8 %tmp495, 11		; <i1> [#uses=1]
	%tmp496.upgrd.2 = zext i1 %tmp496 to i8		; <i8> [#uses=1]
	store i8 %tmp496.upgrd.2, i8* null
	ret void
}
