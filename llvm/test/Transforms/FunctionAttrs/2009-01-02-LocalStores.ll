; RUN: llvm-as < %s | opt -functionattrs | llvm-dis | not grep {nocapture *%%q}
; RUN: llvm-as < %s | opt -functionattrs | llvm-dis | grep {nocapture *%%p}

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
