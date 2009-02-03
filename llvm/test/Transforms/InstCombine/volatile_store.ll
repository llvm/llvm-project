; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {volatile store}
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {volatile load}

@x = weak global i32 0		; <i32*> [#uses=2]

define void @self_assign_1() {
entry:
	%tmp = volatile load i32* @x		; <i32> [#uses=1]
	volatile store i32 %tmp, i32* @x
	br label %return

return:		; preds = %entry
	ret void
}
