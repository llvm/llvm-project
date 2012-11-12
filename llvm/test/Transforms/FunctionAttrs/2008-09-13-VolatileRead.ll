; RUN: opt < %s -functionattrs -S | not grep read
; PR2792

@g = global i32 0		; <i32*> [#uses=1]

define i32 @f() {
	%t = load volatile i32* @g		; <i32> [#uses=1]
	ret i32 %t
}
