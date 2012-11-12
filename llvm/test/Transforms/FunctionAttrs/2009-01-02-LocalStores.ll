; RUN: opt < %s -functionattrs -S | not grep "nocapture *%%q"
; RUN: opt < %s -functionattrs -S | grep "nocapture *%%p"

define i32* @a(i32** %p) {
	%tmp = load i32** %p
	ret i32* %tmp
}

define i32* @b(i32 *%q) {
	%mem = alloca i32*
	store i32* %q, i32** %mem
	%tmp = call i32* @a(i32** %mem)
	ret i32* %tmp
}
