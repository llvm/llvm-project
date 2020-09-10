; RUN: opt -S -basic-aa -dse < %s | FileCheck %s

target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) nounwind
declare void @llvm.memset.p0i8.i8(i8* nocapture, i8, i8, i1) nounwind

define void @test1() {
; CHECK-LABEL: @test1(
  %A = alloca i8

  store i8 0, i8* %A  ;; Written to by memset
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %A)
; CHECK: lifetime.end

  call void @llvm.memset.p0i8.i8(i8* %A, i8 0, i8 -1, i1 false)
; CHECK-NOT: memset

  ret void
; CHECK: ret void
}

define void @test2(i32* %P) {
; CHECK: test2
  %Q = getelementptr i32, i32* %P, i32 1
  %R = bitcast i32* %Q to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %R)
; CHECK: lifetime.start
  store i32 0, i32* %Q  ;; This store is dead.
; CHECK-NOT: store
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %R)
; CHECK: lifetime.end
  ret void
}
