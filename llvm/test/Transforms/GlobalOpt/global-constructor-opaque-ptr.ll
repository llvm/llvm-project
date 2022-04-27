; RUN: opt -passes=globalopt -S < %s | FileCheck %s

; CHECK: @f1
; CHECK: @f2

@llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @f1, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @f2, ptr null }]

define void @f1(i32 %args) {
  ret void
}

define i32 @f2(i32 %args) {
  ret i32 0
}
