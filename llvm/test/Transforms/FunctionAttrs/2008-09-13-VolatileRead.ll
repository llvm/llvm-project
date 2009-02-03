; RUN: llvm-as < %s | opt -functionattrs | llvm-dis | not grep read
; PR2792

@g = global i32 0		; <i32*> [#uses=1]

define i32 @f() {
	%t = volatile load i32* @g		; <i32> [#uses=1]
	ret i32 %t
}
