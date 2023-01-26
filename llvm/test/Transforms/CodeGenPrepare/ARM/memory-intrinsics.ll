; RUN: opt -codegenprepare -mtriple=arm7-unknown-unknown -S < %s | FileCheck %s

declare void @llvm.memcpy.p0.p0.i32(ptr, ptr, i32, i1) nounwind
declare void @llvm.memmove.p0.p0.i32(ptr, ptr, i32, i1) nounwind
declare void @llvm.memset.p0.i32(ptr, i8, i32, i1) nounwind

define void @test_memcpy(ptr align 4 %dst, ptr align 8 %src, i32 %N) {
; CHECK-LABEL: @test_memcpy
; CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 4 %dst, ptr align 8 %src, i32 %N, i1 false)
; CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 4 %dst, ptr align 8 %src, i32 %N, i1 false)
; CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 8 %dst, ptr align 16 %src, i32 %N, i1 false)
entry:
  call void @llvm.memcpy.p0.p0.i32(ptr %dst, ptr %src, i32 %N, i1 false)
  call void @llvm.memcpy.p0.p0.i32(ptr align 2 %dst, ptr align 2 %src, i32 %N, i1 false)
  call void @llvm.memcpy.p0.p0.i32(ptr align 8 %dst, ptr align 16 %src, i32 %N, i1 false)
  ret void
}

define void @test_memmove(ptr align 4 %dst, ptr align 8 %src, i32 %N) {
; CHECK-LABEL: @test_memmove
; CHECK: call void @llvm.memmove.p0.p0.i32(ptr align 4 %dst, ptr align 8 %src, i32 %N, i1 false)
; CHECK: call void @llvm.memmove.p0.p0.i32(ptr align 4 %dst, ptr align 8 %src, i32 %N, i1 false)
; CHECK: call void @llvm.memmove.p0.p0.i32(ptr align 8 %dst, ptr align 16 %src, i32 %N, i1 false)
entry:
  call void @llvm.memmove.p0.p0.i32(ptr %dst, ptr %src, i32 %N, i1 false)
  call void @llvm.memmove.p0.p0.i32(ptr align 2 %dst, ptr align 2 %src, i32 %N, i1 false)
  call void @llvm.memmove.p0.p0.i32(ptr align 8 %dst, ptr align 16 %src, i32 %N, i1 false)
  ret void
}

define void @test_memset(ptr align 4 %dst, i8 %val, i32 %N) {
; CHECK-LABEL: @test_memset
; CHECK: call void @llvm.memset.p0.i32(ptr align 4 %dst, i8 %val, i32 %N, i1 false)
; CHECK: call void @llvm.memset.p0.i32(ptr align 4 %dst, i8 %val, i32 %N, i1 false)
; CHECK: call void @llvm.memset.p0.i32(ptr align 8 %dst, i8 %val, i32 %N, i1 false)
entry:
  call void @llvm.memset.p0.i32(ptr %dst, i8 %val, i32 %N, i1 false)
  call void @llvm.memset.p0.i32(ptr align 2 %dst, i8 %val, i32 %N, i1 false)
  call void @llvm.memset.p0.i32(ptr align 8 %dst, i8 %val, i32 %N, i1 false)
  ret void
}


