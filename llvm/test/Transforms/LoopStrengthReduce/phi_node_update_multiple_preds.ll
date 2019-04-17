; RUN: opt < %s -loop-reduce -disable-output
; LSR should not crash on this.

define fastcc void @loadloop() {
entry:
	switch i8 0, label %shortcirc_next [
		 i8 32, label %loopexit.2
		 i8 59, label %loopexit.2
	]
shortcirc_next:		; preds = %no_exit.2, %entry
	%indvar37 = phi i32 [ 0, %entry ], [ %indvar.next38, %no_exit.2 ]		; <i32> [#uses=3]
	%gep.upgrd.1 = zext i32 %indvar37 to i64		; <i64> [#uses=1]
	%wp.2.4 = getelementptr i8, i8* null, i64 %gep.upgrd.1		; <i8*> [#uses=1]
	br i1 false, label %loopexit.2, label %no_exit.2
no_exit.2:		; preds = %shortcirc_next
	%wp.2.4.rec = bitcast i32 %indvar37 to i32		; <i32> [#uses=1]
	%inc.1.rec = add i32 %wp.2.4.rec, 1		; <i32> [#uses=1]
	%inc.1 = getelementptr i8, i8* null, i32 %inc.1.rec		; <i8*> [#uses=2]
	%indvar.next38 = add i32 %indvar37, 1		; <i32> [#uses=1]
	switch i8 0, label %shortcirc_next [
		 i8 32, label %loopexit.2
		 i8 59, label %loopexit.2
	]
loopexit.2:		; preds = %no_exit.2, %no_exit.2, %shortcirc_next, %entry, %entry
	%wp.2.7 = phi i8* [ null, %entry ], [ null, %entry ], [ %wp.2.4, %shortcirc_next ], [ %inc.1, %no_exit.2 ], [ %inc.1, %no_exit.2 ]		; <i8*> [#uses=0]
	ret void
}

