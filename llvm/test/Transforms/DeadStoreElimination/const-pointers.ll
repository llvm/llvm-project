; RUN: opt -passes=dse -S < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%t = type { i32 }

@g = global i32 42

define void @test1(ptr noalias %pp) {

  store i32 1, ptr %pp; <-- This is dead
  %x = load i32, ptr inttoptr (i32 12345 to ptr)
  store i32 %x, ptr %pp
  ret void
; CHECK-LABEL: define void @test1(
; CHECK: store
; CHECK-NOT: store
; CHECK: ret void
}

define void @test3() {
  store i32 1, ptr @g; <-- This is dead.
  store i32 42, ptr @g
  ret void
; CHECK-LABEL: define void @test3(
; CHECK: store
; CHECK-NOT: store
; CHECK: ret void
}

define void @test4(ptr %p) {
  store i32 1, ptr %p
  %x = load i32, ptr @g; <-- %p and @g could alias
  store i32 %x, ptr %p
  ret void
; CHECK-LABEL: define void @test4(
; CHECK: store
; CHECK: store
; CHECK: ret void
}
