; RUN: llc -verify-machineinstrs < %s -mtriple=ppc64--
; PR1596

	%struct._obstack_chunk = type { i8* }
	%struct.obstack = type { i8*, %struct._obstack_chunk* (i8*, i64)*, i8*, i8 }

define i32 @_obstack_newchunk(%struct.obstack* %h, i32 %length) {
entry:
	br i1 false, label %cond_false, label %cond_true

cond_true:		; preds = %entry
	br i1 false, label %cond_true28, label %cond_next30

cond_false:		; preds = %entry
	%tmp22 = tail call %struct._obstack_chunk* null( i64 undef )		; <%struct._obstack_chunk*> [#uses=2]
	br i1 false, label %cond_true28, label %cond_next30

cond_true28:		; preds = %cond_false, %cond_true
	%iftmp.0.043.0 = phi %struct._obstack_chunk* [ null, %cond_true ], [ %tmp22, %cond_false ]		; <%struct._obstack_chunk*> [#uses=1]
	tail call void null( )
	br label %cond_next30

cond_next30:		; preds = %cond_true28, %cond_false, %cond_true
	%iftmp.0.043.1 = phi %struct._obstack_chunk* [ %iftmp.0.043.0, %cond_true28 ], [ null, %cond_true ], [ %tmp22, %cond_false ]		; <%struct._obstack_chunk*> [#uses=1]
	%tmp41 = getelementptr %struct._obstack_chunk, %struct._obstack_chunk* %iftmp.0.043.1, i32 0, i32 0		; <i8**> [#uses=1]
	store i8* null, i8** %tmp41, align 8
	ret i32 undef
}
