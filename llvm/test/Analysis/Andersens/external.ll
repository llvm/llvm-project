; RUN: llvm-as < %s | opt -anders-aa -gvn -deadargelim | llvm-dis | grep store | not grep null

; Because the 'internal' function is passed to an external function, we don't
; know what the incoming values will alias.  As such, we cannot do the 
; optimization checked by the 'arg-must-alias.ll' test.

declare void @external(i32(i32*)*)
@G = internal constant i32* null

define internal i32 @internal(i32* %ARG) {
	;;; We *DON'T* know that ARG always points to null!
	store i32* %ARG, i32** @G
	ret i32 0
}

define i32 @foo() {
	call void @external(i32(i32*)* @internal)
	%V = call i32 @internal(i32* null)
	ret i32 %V
}
