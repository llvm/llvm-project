; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: 'allockind()' requires exactly one of alloc, realloc, and free
declare i8* @a(i32) allockind("aligned")

; CHECK: 'allockind()' requires exactly one of alloc, realloc, and free
declare i8* @b(i32*) allockind("free,realloc")

; CHECK: 'allockind("free")' doesn't allow uninitialized, zeroed, or aligned modifiers.
declare i8* @c(i32) allockind("free,zeroed")

; CHECK: 'allockind()' can't be both zeroed and uninitialized
declare i8* @d(i32, i32*) allockind("realloc,uninitialized,zeroed")

; CHECK: 'allockind()' requires exactly one of alloc, realloc, and free
declare i8* @e(i32, i32) allockind("alloc,free")
