; RUN: opt < %s -instcombine -S | grep "store volatile"
; RUN: opt < %s -instcombine -S | grep "load volatile"

@x = weak global i32 0		; <i32*> [#uses=2]

define void @self_assign_1() {
entry:
	%tmp = load volatile i32* @x		; <i32> [#uses=1]
	store volatile i32 %tmp, i32* @x
	br label %return

return:		; preds = %entry
	ret void
}
