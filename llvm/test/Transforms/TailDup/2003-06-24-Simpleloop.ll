; RUN: llvm-as < %s | opt -tailduplicate -disable-output

define void @motion_result7() {
entry:
	br label %endif
endif:		; preds = %no_exit, %entry
	%i.1 = phi i32 [ %inc, %no_exit ], [ 0, %entry ]		; <i32> [#uses=1]
	%inc = add i32 %i.1, 1		; <i32> [#uses=1]
	br i1 false, label %no_exit, label %UnifiedExitNode
no_exit:		; preds = %endif
	br i1 false, label %UnifiedExitNode, label %endif
UnifiedExitNode:		; preds = %no_exit, %endif
	ret void
}

