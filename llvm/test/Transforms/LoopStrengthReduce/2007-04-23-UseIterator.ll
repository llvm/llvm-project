; RUN: llvm-as < %s | opt -loop-reduce -disable-output

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"

target triple = "i686-apple-darwin9"

define i8* @foo( i8* %ABC) {
entry:
	switch i8 0, label %bb129 [
		 i8 0, label %UnifiedReturnBlock
		 i8 9, label %UnifiedReturnBlock
		 i8 32, label %UnifiedReturnBlock
		 i8 35, label %UnifiedReturnBlock
		 i8 37, label %bb16.preheader
	]

bb16.preheader:		; preds = %entry
	br label %bb16

bb16:		; preds = %cond_next102, %bb16.preheader
	%indvar = phi i32 [ %indvar.next, %cond_next102 ], [ 0, %bb16.preheader ]		; <i32> [#uses=2]
	%ABC.2146.0.rec = mul i32 %indvar, 3		; <i32> [#uses=1]
	br i1 false, label %UnifiedReturnBlock.loopexit, label %cond_next102

cond_next102:		; preds = %bb16
	%tmp138145.rec = add i32 %ABC.2146.0.rec, 3		; <i32> [#uses=1]
	%tmp138145 = getelementptr i8* %ABC, i32 %tmp138145.rec		; <i8*> [#uses=4]
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=1]
	switch i8 0, label %bb129.loopexit [
		 i8 0, label %UnifiedReturnBlock.loopexit
		 i8 9, label %UnifiedReturnBlock.loopexit
		 i8 32, label %UnifiedReturnBlock.loopexit
		 i8 35, label %UnifiedReturnBlock.loopexit
		 i8 37, label %bb16
	]

bb129.loopexit:		; preds = %cond_next102
	br label %bb129

bb129:		; preds = %bb129.loopexit, %entry
	ret i8* null

UnifiedReturnBlock.loopexit:		; preds = %cond_next102, %cond_next102, %cond_next102, %cond_next102, %bb16
	%UnifiedRetVal.ph = phi i8* [ %tmp138145, %cond_next102 ], [ %tmp138145, %cond_next102 ], [ %tmp138145, %cond_next102 ], [ %tmp138145, %cond_next102 ], [ null, %bb16 ]		; <i8*> [#uses=0]
	br label %UnifiedReturnBlock

UnifiedReturnBlock:		; preds = %UnifiedReturnBlock.loopexit, %entry, %entry, %entry, %entry
	ret i8* null
}

define i8* @bar() {
entry:
	switch i8 0, label %bb158 [
		 i8 37, label %bb74
		 i8 58, label %cond_true
		 i8 64, label %bb11
	]

bb11:		; preds = %entry
	ret i8* null

cond_true:		; preds = %entry
	ret i8* null

bb74:		; preds = %entry
	ret i8* null

bb158:		; preds = %entry
	ret i8* null
}

