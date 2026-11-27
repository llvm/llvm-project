; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

; CHECK: 'allockind()' requires exactly one of alloc, realloc, and free
declare ptr @a(i32) allockind("aligned")

; CHECK: 'allockind()' requires exactly one of alloc, realloc, and free
declare ptr @b(ptr) allockind("free,realloc")

; CHECK: 'allockind("free")' doesn't allow uninitialized, zeroed, or aligned modifiers.
declare void @c(ptr) allockind("free,zeroed")

; CHECK: 'allockind()' can't be both zeroed and uninitialized
declare ptr @d(i32, ptr) allockind("realloc,uninitialized,zeroed")

; CHECK: 'allockind()' requires exactly one of alloc, realloc, and free
declare ptr @e(i32, i32) allockind("alloc,free")

; CHECK: 'allockind("free")' requires a void or integer return type
declare ptr @f(ptr allocptr) allockind("free")

; CHECK: 'allockind("free")' requires a void or integer return type
define void @g(ptr %p) {
  %r = call ptr @h(ptr %p) allockind("free")
  ret void
}
declare ptr @h(ptr)

; CHECK-NOT: allockind
declare void @i(ptr allocptr) allockind("free")
declare i32 @j(ptr allocptr) allockind("free")
