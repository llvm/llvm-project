; Exit blocks need to be updated for all nested loops...

; RUN: llvm-as < %s | opt -loopsimplify

define i32 @yyparse() {
bb0:
	br i1 false, label %UnifiedExitNode, label %bb19
bb19:		; preds = %bb28, %bb0
	br i1 false, label %bb28, label %UnifiedExitNode
bb28:		; preds = %bb32, %bb19
	br i1 false, label %bb32, label %bb19
bb32:		; preds = %bb28
	br i1 false, label %UnifiedExitNode, label %bb28
UnifiedExitNode:		; preds = %bb32, %bb19, %bb0
	ret i32 0
}

