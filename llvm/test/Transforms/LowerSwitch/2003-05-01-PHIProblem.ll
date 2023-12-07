; RUN: opt < %s -passes=lowerswitch

define void @child(i32 %ct.1) {
entry:
	switch i32 0, label %return [
		 i32 3, label %UnifiedExitNode
		 i32 0, label %return
	]
return:		; preds = %entry, %entry
	%result.0 = phi ptr [ null, %entry ], [ null, %entry ]		; <ptr> [#uses=0]
	br label %UnifiedExitNode
UnifiedExitNode:		; preds = %return, %entry
	ret void
}

