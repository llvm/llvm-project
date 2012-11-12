; RUN: opt -globalopt -S -o - < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare {}* @llvm.invariant.start(i64 %size, i8* nocapture %ptr)

define void @test1(i8* %ptr) {
  call {}* @llvm.invariant.start(i64 4, i8* %ptr)
  ret void
}

@object1 = global i32 0
; CHECK: @object1 = constant i32 -1
define void @ctor1() {
  store i32 -1, i32* @object1
  %A = bitcast i32* @object1 to i8*
  call void @test1(i8* %A)
  ret void
}


@object2 = global i32 0
; CHECK: @object2 = global i32 0
define void @ctor2() {
  store i32 -1, i32* @object2
  %A = bitcast i32* @object2 to i8*
  %B = call {}* @llvm.invariant.start(i64 4, i8* %A)
  %C = bitcast {}* %B to i8*
  ret void
}


@object3 = global i32 0
; CHECK: @object3 = global i32 -1
define void @ctor3() {
  store i32 -1, i32* @object3
  %A = bitcast i32* @object3 to i8*
  call {}* @llvm.invariant.start(i64 3, i8* %A)
  ret void
}


@object4 = global i32 0
; CHECK: @object4 = global i32 -1
define void @ctor4() {
  store i32 -1, i32* @object4
  %A = bitcast i32* @object4 to i8*
  call {}* @llvm.invariant.start(i64 -1, i8* %A)
  ret void
}


@llvm.global_ctors = appending constant
  [4 x { i32, void ()* }]
  [ { i32, void ()* } { i32 65535, void ()* @ctor1 },
    { i32, void ()* } { i32 65535, void ()* @ctor2 },
    { i32, void ()* } { i32 65535, void ()* @ctor3 },
    { i32, void ()* } { i32 65535, void ()* @ctor4 } ]
