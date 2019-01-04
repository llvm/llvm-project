; RUN: llvm-as < %s | llvm-dis | not grep ptrtoint
; RUN: verify-uselistorder %s
; All of these should be eliminable


define i32 @foo() {
	ret i32 and (i32 ptrtoint (i32()* @foo to i32), i32 1)
}

define i32 @foo2() {
	ret i32 and (i32 1, i32 ptrtoint (i32()* @foo2 to i32))
}

define i1 @foo3() {
	ret i1 icmp ne (i1()* @foo3, i1()* null)
}
