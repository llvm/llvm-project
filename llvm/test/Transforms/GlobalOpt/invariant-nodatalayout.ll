; RUN: opt -globalopt -S -o - < %s | FileCheck %s
; The check here is that it doesn't crash.

declare {}* @llvm.invariant.start.p0i8(i64 %size, i8* nocapture %ptr)

@object1 = global { i32, i32 } zeroinitializer
; CHECK: @object1 = global { i32, i32 } zeroinitializer

define void @ctor1() {
  %ptr = bitcast {i32, i32}* @object1 to i8*
  call {}* @llvm.invariant.start.p0i8(i64 4, i8* %ptr)
  ret void
}

@llvm.global_ctors = appending constant
  [1 x { i32, void ()* }]
  [ { i32, void ()* } { i32 65535, void ()* @ctor1 } ]
