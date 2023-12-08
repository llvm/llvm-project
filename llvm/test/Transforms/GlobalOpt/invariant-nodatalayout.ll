; RUN: opt -passes=globalopt -S -o - < %s | FileCheck %s
; The check here is that it doesn't crash.

declare ptr @llvm.invariant.start.p0(i64 %size, ptr nocapture %ptr)

@object1 = global { i32, i32 } zeroinitializer
; CHECK: @object1 = global { i32, i32 } zeroinitializer

define void @ctor1() {
  call ptr @llvm.invariant.start.p0(i64 4, ptr @object1)
  ret void
}

@llvm.global_ctors = appending constant
  [1 x { i32, ptr, ptr }]
  [ { i32, ptr, ptr } { i32 65535, ptr @ctor1, ptr null } ]
