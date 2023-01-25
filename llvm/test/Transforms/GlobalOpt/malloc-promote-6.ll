; RUN: opt -passes=globalopt -S < %s | FileCheck %s

; CHECK-NOT: @global

@global = internal global ptr null
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @zot, ptr null }]

declare ptr @_Znwm(i64)

define internal void @widget() {
  %tmp = tail call ptr @_Znwm(i64 0)
  store ptr %tmp, ptr @global, align 8
  call void @baz(ptr @spam)
  ret void
}

define internal void @spam() {
  %tmp = load ptr, ptr @global, align 8
  ret void
}

define internal void @zot() {
  call void @baz(ptr @widget)
  ret void
}

declare void @baz(ptr)

