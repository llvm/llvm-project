; RUN: opt < %s -basicaa -memcpyopt -S | FileCheck %s

target datalayout = "e"

declare void @foo(i8*)
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind

define void @test() {
  %ptr1 = alloca i8
  %ptr2 = alloca i8
  call void @foo(i8* %ptr2)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %ptr1, i8* %ptr2, i32 1, i32 1, i1 false)
  call void @foo(i8* %ptr1)
  ret void

  ; Check that the transformation isn't applied if the called function can
  ; capture the pointer argument (i.e. the nocapture attribute isn't present)
  ; CHECK-LABEL: @test(
  ; CHECK: call void @foo(i8* %ptr2)
  ; CHECK-NEXT: call void @llvm.memcpy
  ; CHECK-NEXT: call void @foo(i8* %ptr1)
}
