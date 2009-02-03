;RUN: llvm-as < %s | opt -simplifycfg -disable-output
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

define i32 @bork() nounwind  {
entry:
	br label %bb5.outer

bb5.outer.loopexit:		; preds = %bb5
	br label %bb5.outer

bb5.outer:		; preds = %bb5.outer.loopexit, %entry
	%undo.0.ph = phi i32 [ 0, %entry ], [ 1, %bb5.outer.loopexit ]		; <i32> [#uses=1]
	br label %bb5

bb5:		; preds = %bb5, %bb5.outer
	%tmp6 = tail call i32 (...)* @foo( ) nounwind 		; <i32> [#uses=1]
	switch i32 %tmp6, label %bb13 [
		 i32 -1, label %bb10
		 i32 102, label %bb5
		 i32 110, label %bb5.outer.loopexit
	]

bb10:		; preds = %bb5
	%tmp12 = tail call i32 (...)* @bar( i32 %undo.0.ph ) nounwind 		; <i32> [#uses=0]
	br label %UnifiedReturnBlock

bb13:		; preds = %bb5
	br label %UnifiedReturnBlock

UnifiedReturnBlock:		; preds = %bb13, %bb10
	%UnifiedRetVal = phi i32 [ 1, %bb10 ], [ 258, %bb13 ]		; <i32> [#uses=1]
	ret i32 %UnifiedRetVal
}

declare i32 @foo(...)

declare i32 @bar(...)
