; RUN: opt -S -dse < %s | FileCheck %s

declare void @llvm.memcpy.p0i8.p0i8.i8(i8* nocapture, i8* nocapture, i8, i32, i1) nounwind
declare void @llvm.memmove.p0i8.p0i8.i8(i8* nocapture, i8* nocapture, i8, i32, i1) nounwind
declare void @llvm.memset.p0i8.i8(i8* nocapture, i8, i8, i32, i1) nounwind

define void @test1() {
; CHECK: @test1
  %A = alloca i8
  %B = alloca i8

  store i8 0, i8* %A  ;; Written to by memcpy
; CHECK-NOT: store

  call void @llvm.memcpy.p0i8.p0i8.i8(i8* %A, i8* %B, i8 -1, i32 0, i1 false)

  ret void
; CHECK: ret void
}

define void @test2() {
; CHECK: @test2
  %A = alloca i8
  %B = alloca i8

  store i8 0, i8* %A  ;; Written to by memmove
; CHECK-NOT: store

  call void @llvm.memmove.p0i8.p0i8.i8(i8* %A, i8* %B, i8 -1, i32 0, i1 false)

  ret void
; CHECK: ret void
}

define void @test3() {
; CHECK: @test3
  %A = alloca i8
  %B = alloca i8

  store i8 0, i8* %A  ;; Written to by memset
; CHECK-NOT: store

  call void @llvm.memset.p0i8.i8(i8* %A, i8 0, i8 -1, i32 0, i1 false)

  ret void
; CHECK: ret void
}
