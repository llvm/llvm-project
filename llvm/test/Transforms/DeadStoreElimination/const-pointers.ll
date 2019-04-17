; RUN: opt -basicaa -dse -S < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%t = type { i32 }

@g = global i32 42

define void @test1(%t* noalias %pp) {
  %p = getelementptr inbounds %t, %t* %pp, i32 0, i32 0

  store i32 1, i32* %p; <-- This is dead
  %x = load i32, i32* inttoptr (i32 12345 to i32*)
  store i32 %x, i32* %p
  ret void
; CHECK-LABEL: define void @test1(
; CHECK: store
; CHECK-NOT: store
; CHECK: ret void
}

define void @test3() {
  store i32 1, i32* @g; <-- This is dead.
  store i32 42, i32* @g
  ret void
; CHECK-LABEL: define void @test3(
; CHECK: store
; CHECK-NOT: store
; CHECK: ret void
}

define void @test4(i32* %p) {
  store i32 1, i32* %p
  %x = load i32, i32* @g; <-- %p and @g could alias
  store i32 %x, i32* %p
  ret void
; CHECK-LABEL: define void @test4(
; CHECK: store
; CHECK: store
; CHECK: ret void
}
