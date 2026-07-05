; RUN: llc < %s
; Infinite loop in the dag combiner, reduced from 176.gcc.	
%struct._obstack_chunk = type { ptr, ptr, [4 x i8] }
	%struct.anon = type { i32 }
	%struct.lang_decl = type opaque
	%struct.lang_type = type { i32, [1 x ptr] }
	%struct.obstack = type { i32, ptr, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, i8 }
	%struct.rtx_def = type { i16, i8, i8, [1 x %struct.anon] }
	%struct.tree_common = type { ptr, ptr, i8, i8, i8, i8 }
	%struct.tree_decl = type { [12 x i8], ptr, i32, ptr, i32, i8, i8, i8, i8, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, %struct.anon, { ptr }, ptr, ptr }
	%struct.tree_list = type { [12 x i8], ptr, ptr }
	%struct.tree_node = type { %struct.tree_decl }
	%struct.tree_type = type { [12 x i8], ptr, ptr, ptr, i32, i8, i8, i8, i8, i32, ptr, ptr, %struct.anon, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
@void_type_node = external global ptr		; <ptr> [#uses=1]
@char_type_node = external global ptr		; <ptr> [#uses=1]
@short_integer_type_node = external global ptr		; <ptr> [#uses=1]
@short_unsigned_type_node = external global ptr		; <ptr> [#uses=1]
@float_type_node = external global ptr		; <ptr> [#uses=1]
@signed_char_type_node = external global ptr		; <ptr> [#uses=1]
@unsigned_char_type_node = external global ptr		; <ptr> [#uses=1]

define fastcc i32 @self_promoting_args_p(ptr %parms) {
entry:
	%tmp915 = icmp eq ptr %parms, null		; <i1> [#uses=1]
	br i1 %tmp915, label %return, label %cond_true92.preheader

cond_true:		; preds = %cond_true92
	%tmp9.not = icmp ne ptr %tmp2, %tmp7		; <i1> [#uses=1]
	%tmp14 = icmp eq ptr %tmp2, null		; <i1> [#uses=1]
	%bothcond = or i1 %tmp9.not, %tmp14		; <i1> [#uses=1]
	br i1 %bothcond, label %return, label %cond_next18

cond_next12:		; preds = %cond_true92
	%tmp14.old = icmp eq ptr %tmp2, null		; <i1> [#uses=1]
	br i1 %tmp14.old, label %return, label %cond_next18

cond_next18:		; preds = %cond_next12, %cond_true
	%tmp21 = getelementptr %struct.tree_type, ptr %tmp2, i32 0, i32 17		; <ptr> [#uses=1]
	%tmp22 = load ptr, ptr %tmp21		; <ptr> [#uses=6]
	%tmp24 = icmp eq ptr %tmp22, %tmp23		; <i1> [#uses=1]
	br i1 %tmp24, label %return, label %cond_next28

cond_next28:		; preds = %cond_next18
	%tmp = getelementptr %struct.tree_common, ptr %tmp2, i32 0, i32 2		; <ptr> [#uses=1]
	%tmp.upgrd.2 = load i32, ptr %tmp		; <i32> [#uses=1]
	%tmp32 = trunc i32 %tmp.upgrd.2 to i8		; <i8> [#uses=1]
	%tmp33 = icmp eq i8 %tmp32, 7		; <i1> [#uses=1]
	br i1 %tmp33, label %cond_true34, label %cond_next84

cond_true34:		; preds = %cond_next28
	%tmp40 = icmp eq ptr %tmp22, %tmp39		; <i1> [#uses=1]
	%tmp49 = icmp eq ptr %tmp22, %tmp48		; <i1> [#uses=1]
	%bothcond6 = or i1 %tmp40, %tmp49		; <i1> [#uses=1]
	%tmp58 = icmp eq ptr %tmp22, %tmp57		; <i1> [#uses=1]
	%bothcond7 = or i1 %bothcond6, %tmp58		; <i1> [#uses=1]
	%tmp67 = icmp eq ptr %tmp22, %tmp66		; <i1> [#uses=1]
	%bothcond8 = or i1 %bothcond7, %tmp67		; <i1> [#uses=1]
	%tmp76 = icmp eq ptr %tmp22, %tmp75		; <i1> [#uses=1]
	%bothcond9 = or i1 %bothcond8, %tmp76		; <i1> [#uses=2]
	%brmerge = or i1 %bothcond9, %tmp.upgrd.6		; <i1> [#uses=1]
	%bothcond9.upgrd.3 = zext i1 %bothcond9 to i32		; <i32> [#uses=1]
	%.mux = xor i32 %bothcond9.upgrd.3, 1		; <i32> [#uses=1]
	br i1 %brmerge, label %return, label %cond_true92

cond_next84:		; preds = %cond_next28
	br i1 %tmp.upgrd.6, label %return, label %cond_true92

cond_true92.preheader:		; preds = %entry
	%tmp7 = load ptr, ptr @void_type_node		; <ptr> [#uses=1]
	%tmp23 = load ptr, ptr @float_type_node		; <ptr> [#uses=1]
	%tmp39 = load ptr, ptr @char_type_node		; <ptr> [#uses=1]
	%tmp48 = load ptr, ptr @signed_char_type_node		; <ptr> [#uses=1]
	%tmp57 = load ptr, ptr @unsigned_char_type_node		; <ptr> [#uses=1]
	%tmp66 = load ptr, ptr @short_integer_type_node		; <ptr> [#uses=1]
	%tmp75 = load ptr, ptr @short_unsigned_type_node		; <ptr> [#uses=1]
	br label %cond_true92

cond_true92:		; preds = %cond_true92.preheader, %cond_next84, %cond_true34
	%t.0.0 = phi ptr [ %parms, %cond_true92.preheader ], [ %tmp6, %cond_true34 ], [ %tmp6, %cond_next84 ]		; <ptr> [#uses=2]
	%tmp.upgrd.5 = getelementptr %struct.tree_list, ptr %t.0.0, i32 0, i32 2		; <ptr> [#uses=1]
	%tmp2 = load ptr, ptr %tmp.upgrd.5		; <ptr> [#uses=5]
	%tmp5 = getelementptr %struct.tree_common, ptr %t.0.0, i32 0, i32 0		; <ptr> [#uses=1]
	%tmp6 = load ptr, ptr %tmp5		; <ptr> [#uses=3]
	%tmp.upgrd.6 = icmp eq ptr %tmp6, null		; <i1> [#uses=3]
	br i1 %tmp.upgrd.6, label %cond_true, label %cond_next12

return:		; preds = %cond_next84, %cond_true34, %cond_next18, %cond_next12, %cond_true, %entry
	%retval.0 = phi i32 [ 1, %entry ], [ 1, %cond_next84 ], [ %.mux, %cond_true34 ], [ 0, %cond_next18 ], [ 0, %cond_next12 ], [ 0, %cond_true ]		; <i32> [#uses=1]
	ret i32 %retval.0
}
