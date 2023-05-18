; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: 'allockind()' requires exactly one of alloc, realloc, and free
declare ptr @a(i32) allockind("aligned")

; CHECK: 'allockind()' requires exactly one of alloc, realloc, and free
declare ptr @b(ptr) allockind("free,realloc")

; CHECK: 'allockind("free")' doesn't allow uninitialized, zeroed, or aligned modifiers.
declare ptr @c(i32) allockind("free,zeroed")

; CHECK: 'allockind()' can't be both zeroed and uninitialized
declare ptr @d(i32, ptr) allockind("realloc,uninitialized,zeroed")

; CHECK: 'allockind()' requires exactly one of alloc, realloc, and free
declare ptr @e(i32, i32) allockind("alloc,free")
