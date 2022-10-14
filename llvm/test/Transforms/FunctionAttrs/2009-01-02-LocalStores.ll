; RUN: opt < %s -function-attrs -S | FileCheck %s
; RUN: opt < %s -passes=function-attrs -S | FileCheck %s

; CHECK: define ptr @a(ptr nocapture readonly %p)
define ptr @a(ptr %p) {
	%tmp = load ptr, ptr %p
	ret ptr %tmp
}

; CHECK: define ptr @b(ptr %q)
define ptr @b(ptr %q) {
	%mem = alloca ptr
	store ptr %q, ptr %mem
	%tmp = call ptr @a(ptr %mem)
	ret ptr %tmp
}

; CHECK: define ptr @c(ptr readnone returned %r)
@g = global i32 0
define ptr @c(ptr %r) {
	%a = icmp eq ptr %r, null
	store i32 1, ptr @g
	ret ptr %r
}
