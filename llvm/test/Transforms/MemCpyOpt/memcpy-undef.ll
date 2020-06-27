; RUN: opt < %s -basic-aa -memcpyopt -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

%struct.foo = type { i8, [7 x i8], i32 }

define i32 @test1(%struct.foo* nocapture %foobie) nounwind noinline ssp uwtable {
  %bletch.sroa.1 = alloca [7 x i8], align 1
  %1 = getelementptr inbounds %struct.foo, %struct.foo* %foobie, i64 0, i32 0
  store i8 98, i8* %1, align 4
  %2 = getelementptr inbounds %struct.foo, %struct.foo* %foobie, i64 0, i32 1, i64 0
  %3 = getelementptr inbounds [7 x i8], [7 x i8]* %bletch.sroa.1, i64 0, i64 0
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %2, i8* %3, i64 7, i1 false)
  %4 = getelementptr inbounds %struct.foo, %struct.foo* %foobie, i64 0, i32 2
  store i32 20, i32* %4, align 4
  ret i32 undef

; Check that the memcpy is removed.
; CHECK-LABEL: @test1(
; CHECK-NOT: call void @llvm.memcpy
}

define void @test2(i8* sret noalias nocapture %out, i8* %in) nounwind noinline ssp uwtable {
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %in)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %out, i8* %in, i64 8, i1 false)
  ret void

; Check that the memcpy is removed.
; CHECK-LABEL: @test2(
; CHECK-NOT: call void @llvm.memcpy
}

define void @test3(i8* sret noalias nocapture %out, i8* %in) nounwind noinline ssp uwtable {
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %in)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %out, i8* %in, i64 8, i1 false)
  ret void

; Check that the memcpy is not removed.
; CHECK-LABEL: @test3(
; CHECK: call void @llvm.memcpy
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1) nounwind

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) nounwind
