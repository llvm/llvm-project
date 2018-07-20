; RUN: opt < %s -functionattrs -S | FileCheck %s
; RUN: opt < %s -passes=function-attrs -S | FileCheck %s

; CHECK: define i32* @a(i32** nocapture readonly %p)
define i32* @a(i32** %p) {
	%tmp = load i32*, i32** %p
	ret i32* %tmp
}

; CHECK: define i32* @b(i32* %q)
define i32* @b(i32 *%q) {
	%mem = alloca i32*
	store i32* %q, i32** %mem
	%tmp = call i32* @a(i32** %mem)
	ret i32* %tmp
}

; CHECK: define i32* @c(i32* readnone returned %r)
@g = global i32 0
define i32* @c(i32 *%r) {
	%a = icmp eq i32* %r, null
	store i32 1, i32* @g
	ret i32* %r
}
