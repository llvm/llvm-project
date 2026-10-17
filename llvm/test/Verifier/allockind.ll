; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: 'allockind()' requires exactly one of alloc, realloc, and free
declare ptr @a(i32) allockind("aligned")

; CHECK: 'allockind()' requires exactly one of alloc, realloc, and free
declare ptr @b(ptr) allockind("free,realloc")

; CHECK: 'allockind("free")' doesn't allow uninitialized, zeroed, aligned, address_unpredictable or alloc_disjoint modifiers.
declare ptr @c(i32) allockind("free,zeroed")

; CHECK: 'allockind()' can't be both zeroed and uninitialized
declare ptr @d(i32, ptr) allockind("realloc,uninitialized,zeroed")

; CHECK: 'allockind()' requires exactly one of alloc, realloc, and free
declare ptr @e(i32, i32) allockind("alloc,free")

; CHECK: 'allockind("free")' doesn't allow uninitialized, zeroed, aligned, address_unpredictable or alloc_disjoint modifiers.
declare ptr @f(i32) allockind("free,alloc_disjoint")

; CHECK: 'allockind("poisons_memory")' requires free
declare ptr @g(i32) allockind("alloc,poisons_memory")
