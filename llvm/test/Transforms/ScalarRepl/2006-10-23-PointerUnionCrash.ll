; RUN: llvm-as < %s | opt -scalarrepl -disable-output
; END.
target datalayout = "e-p:32:32"
target triple = "i686-apple-darwin8.7.2"

define void @glgProcessColor() {
entry:
	%source_ptr = alloca i8*, align 4		; <i8**> [#uses=2]
	br i1 false, label %bb1357, label %cond_next583
cond_next583:		; preds = %entry
	ret void
bb1357:		; preds = %entry
	br i1 false, label %bb1365, label %bb27055
bb1365:		; preds = %bb1357
	switch i32 0, label %cond_next10377 [
		 i32 0, label %bb4679
		 i32 1, label %bb4679
		 i32 2, label %bb4679
		 i32 3, label %bb4679
		 i32 4, label %bb5115
		 i32 5, label %bb6651
		 i32 6, label %bb7147
		 i32 7, label %bb8683
		 i32 8, label %bb9131
		 i32 9, label %bb9875
		 i32 10, label %bb4679
		 i32 11, label %bb4859
		 i32 12, label %bb4679
		 i32 16, label %bb10249
	]
bb4679:		; preds = %bb1365, %bb1365, %bb1365, %bb1365, %bb1365, %bb1365
	ret void
bb4859:		; preds = %bb1365
	ret void
bb5115:		; preds = %bb1365
	ret void
bb6651:		; preds = %bb1365
	ret void
bb7147:		; preds = %bb1365
	ret void
bb8683:		; preds = %bb1365
	ret void
bb9131:		; preds = %bb1365
	ret void
bb9875:		; preds = %bb1365
	%source_ptr9884 = bitcast i8** %source_ptr to i8**		; <i8**> [#uses=1]
	%tmp9885 = load i8** %source_ptr9884		; <i8*> [#uses=0]
	ret void
bb10249:		; preds = %bb1365
	%source_ptr10257 = bitcast i8** %source_ptr to i16**		; <i16**> [#uses=1]
	%tmp10258 = load i16** %source_ptr10257		; <i16*> [#uses=0]
	ret void
cond_next10377:		; preds = %bb1365
	ret void
bb27055:		; preds = %bb1357
	ret void
}
