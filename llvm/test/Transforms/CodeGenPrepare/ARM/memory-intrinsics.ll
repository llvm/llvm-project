; RUN: opt -codegenprepare -mtriple=arm7-unknown-unknown -S < %s | FileCheck %s

declare void @llvm.memcpy.p0i8.p0i8.i32(i8*, i8*, i32, i1) nounwind
declare void @llvm.memmove.p0i8.p0i8.i32(i8*, i8*, i32, i1) nounwind
declare void @llvm.memset.p0i8.i32(i8*, i8, i32, i1) nounwind

define void @test_memcpy(i8* align 4 %dst, i8* align 8 %src, i32 %N) {
; CHECK-LABEL: @test_memcpy
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %dst, i8* align 8 %src, i32 %N, i1 false)
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %dst, i8* align 8 %src, i32 %N, i1 false)
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 %dst, i8* align 16 %src, i32 %N, i1 false)
entry:
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %src, i32 %N, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 2 %dst, i8* align 2 %src, i32 %N, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 %dst, i8* align 16 %src, i32 %N, i1 false)
  ret void
}

define void @test_memmove(i8* align 4 %dst, i8* align 8 %src, i32 %N) {
; CHECK-LABEL: @test_memmove
; CHECK: call void @llvm.memmove.p0i8.p0i8.i32(i8* align 4 %dst, i8* align 8 %src, i32 %N, i1 false)
; CHECK: call void @llvm.memmove.p0i8.p0i8.i32(i8* align 4 %dst, i8* align 8 %src, i32 %N, i1 false)
; CHECK: call void @llvm.memmove.p0i8.p0i8.i32(i8* align 8 %dst, i8* align 16 %src, i32 %N, i1 false)
entry:
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %dst, i8* %src, i32 %N, i1 false)
  call void @llvm.memmove.p0i8.p0i8.i32(i8* align 2 %dst, i8* align 2 %src, i32 %N, i1 false)
  call void @llvm.memmove.p0i8.p0i8.i32(i8* align 8 %dst, i8* align 16 %src, i32 %N, i1 false)
  ret void
}

define void @test_memset(i8* align 4 %dst, i8 %val, i32 %N) {
; CHECK-LABEL: @test_memset
; CHECK: call void @llvm.memset.p0i8.i32(i8* align 4 %dst, i8 %val, i32 %N, i1 false)
; CHECK: call void @llvm.memset.p0i8.i32(i8* align 4 %dst, i8 %val, i32 %N, i1 false)
; CHECK: call void @llvm.memset.p0i8.i32(i8* align 8 %dst, i8 %val, i32 %N, i1 false)
entry:
  call void @llvm.memset.p0i8.i32(i8* %dst, i8 %val, i32 %N, i1 false)
  call void @llvm.memset.p0i8.i32(i8* align 2 %dst, i8 %val, i32 %N, i1 false)
  call void @llvm.memset.p0i8.i32(i8* align 8 %dst, i8 %val, i32 %N, i1 false)
  ret void
}


