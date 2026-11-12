; RUN: opt -passes=globalopt -S -o - < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @llvm.invariant.start.p0(i64 %size, ptr nocapture %ptr)

define void @test1(ptr %ptr) {
  call ptr @llvm.invariant.start.p0(i64 4, ptr %ptr)
  ret void
}

@object1 = global i32 0
; CHECK: @object1 = constant i32 -1
define void @ctor1() {
  store i32 -1, ptr @object1
  %A = bitcast ptr @object1 to ptr
  call void @test1(ptr %A)
  ret void
}


@object2 = global i32 0
; CHECK: @object2 = global i32 0
define void @ctor2() {
  store i32 -1, ptr @object2
  %A = bitcast ptr @object2 to ptr
  %B = call ptr @llvm.invariant.start.p0(i64 4, ptr %A)
  %C = bitcast ptr %B to ptr
  ret void
}


@object3 = global i32 0
; CHECK: @object3 = global i32 -1
define void @ctor3() {
  store i32 -1, ptr @object3
  %A = bitcast ptr @object3 to ptr
  call ptr @llvm.invariant.start.p0(i64 3, ptr %A)
  ret void
}


@object4 = global i32 0
; CHECK: @object4 = global i32 -1
define void @ctor4() {
  store i32 -1, ptr @object4
  %A = bitcast ptr @object4 to ptr
  call ptr @llvm.invariant.start.p0(i64 -1, ptr %A)
  ret void
}


@llvm.global_ctors = appending constant
  [4 x { i32, ptr, ptr }]
  [ { i32, ptr, ptr } { i32 65535, ptr @ctor1, ptr null },
    { i32, ptr, ptr } { i32 65535, ptr @ctor2, ptr null },
    { i32, ptr, ptr } { i32 65535, ptr @ctor3, ptr null },
    { i32, ptr, ptr } { i32 65535, ptr @ctor4, ptr null } ]
