; RUN: llc -verify-machineinstrs < %s -mtriple=ppc64--
; PR1596

	%struct._obstack_chunk = type { ptr }
	%struct.obstack = type { ptr, ptr, ptr, i8 }

define i32 @_obstack_newchunk(ptr %h, i32 %length) {
entry:
	br i1 false, label %cond_false, label %cond_true

cond_true:		; preds = %entry
	br i1 false, label %cond_true28, label %cond_next30

cond_false:		; preds = %entry
	%tmp22 = tail call ptr null( i64 undef )		; <ptr> [#uses=2]
	br i1 false, label %cond_true28, label %cond_next30

cond_true28:		; preds = %cond_false, %cond_true
	%iftmp.0.043.0 = phi ptr [ null, %cond_true ], [ %tmp22, %cond_false ]		; <ptr> [#uses=1]
	tail call void null( )
	br label %cond_next30

cond_next30:		; preds = %cond_true28, %cond_false, %cond_true
	%iftmp.0.043.1 = phi ptr [ %iftmp.0.043.0, %cond_true28 ], [ null, %cond_true ], [ %tmp22, %cond_false ]		; <ptr> [#uses=1]
	%tmp41 = getelementptr %struct._obstack_chunk, ptr %iftmp.0.043.1, i32 0, i32 0		; <ptr> [#uses=1]
	store ptr null, ptr %tmp41, align 8
	ret i32 undef
}
