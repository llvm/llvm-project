; RUN: opt -memcpyopt -S %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; CHECK-LABEL: define void @test(
; CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* %dst1, i8 %c, i64 128, i1 false)
; CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* align 8 %dst2, i8 %c, i64 128, i1 false)
; CHECK-NEXT: ret void
define void @test(i8* %dst1, i8* %dst2, i8 %c) {
  call void @llvm.memset.p0i8.i64(i8* %dst1, i8 %c, i64 128, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %dst2, i8* align 8 %dst1, i64 128, i1 false)
  ret void
}

; CHECK-LABEL: define void @test_smaller_memcpy(
; CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* %dst1, i8 %c, i64 128, i1 false)
; CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* %dst2, i8 %c, i64 100, i1 false)
; CHECK-NEXT: ret void
define void @test_smaller_memcpy(i8* %dst1, i8* %dst2, i8 %c) {
  call void @llvm.memset.p0i8.i64(i8* %dst1, i8 %c, i64 128, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst2, i8* %dst1, i64 100, i1 false)
  ret void
}

; CHECK-LABEL: define void @test_smaller_memset(
; CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* %dst1, i8 %c, i64 100, i1 false)
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst2, i8* %dst1, i64 128, i1 false)
; CHECK-NEXT: ret void
define void @test_smaller_memset(i8* %dst1, i8* %dst2, i8 %c) {
  call void @llvm.memset.p0i8.i64(i8* %dst1, i8 %c, i64 100, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst2, i8* %dst1, i64 128, i1 false)
  ret void
}

; CHECK-LABEL: define void @test_align_memset(
; CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* align 8 %dst1, i8 %c, i64 128, i1 false)
; CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* %dst2, i8 %c, i64 128, i1 false)
; CHECK-NEXT: ret void
define void @test_align_memset(i8* %dst1, i8* %dst2, i8 %c) {
  call void @llvm.memset.p0i8.i64(i8* align 8 %dst1, i8 %c, i64 128, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst2, i8* %dst1, i64 128, i1 false)
  ret void
}

; CHECK-LABEL: define void @test_different_types(
; CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* align 8 %dst1, i8 %c, i64 128, i1 false)
; CHECK-NEXT: call void @llvm.memset.p0i8.i32(i8* %dst2, i8 %c, i32 100, i1 false)
; CHECK-NEXT: ret void
define void @test_different_types(i8* %dst1, i8* %dst2, i8 %c) {
  call void @llvm.memset.p0i8.i64(i8* align 8 %dst1, i8 %c, i64 128, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst2, i8* %dst1, i32 100, i1 false)
  ret void
}

; CHECK-LABEL: define void @test_different_types_2(
; CHECK-NEXT: call void @llvm.memset.p0i8.i32(i8* align 8 %dst1, i8 %c, i32 128, i1 false)
; CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* %dst2, i8 %c, i64 100, i1 false)
; CHECK-NEXT: ret void
define void @test_different_types_2(i8* %dst1, i8* %dst2, i8 %c) {
  call void @llvm.memset.p0i8.i32(i8* align 8 %dst1, i8 %c, i32 128, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst2, i8* %dst1, i64 100, i1 false)
  ret void
}

; CHECK-LABEL: define void @test_different_source_gep(
; CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* %dst1, i8 %c, i64 128, i1 false)
; CHECK-NEXT: %p = getelementptr i8, i8* %dst1, i64 64
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst2, i8* %p, i64 64, i1 false)
; CHECK-NEXT: ret void
define void @test_different_source_gep(i8* %dst1, i8* %dst2, i8 %c) {
  call void @llvm.memset.p0i8.i64(i8* %dst1, i8 %c, i64 128, i1 false)
  ; FIXME: We could optimize this as well.
  %p = getelementptr i8, i8* %dst1, i64 64
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst2, i8* %p, i64 64, i1 false)
  ret void
}

; CHECK-LABEL: define void @test_variable_size_1(
; CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* %dst1, i8 %c, i64 %dst1_size, i1 false)
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst2, i8* %dst1, i64 128, i1 false)
; CHECK-NEXT: ret void
define void @test_variable_size_1(i8* %dst1, i64 %dst1_size, i8* %dst2, i8 %c) {
  call void @llvm.memset.p0i8.i64(i8* %dst1, i8 %c, i64 %dst1_size, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst2, i8* %dst1, i64 128, i1 false)
  ret void
}

; CHECK-LABEL: define void @test_variable_size_2(
; CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* %dst1, i8 %c, i64 128, i1 false)
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst2, i8* %dst1, i64 %dst2_size, i1 false)
; CHECK-NEXT: ret void
define void @test_variable_size_2(i8* %dst1, i8* %dst2, i64 %dst2_size, i8 %c) {
  call void @llvm.memset.p0i8.i64(i8* %dst1, i8 %c, i64 128, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst2, i8* %dst1, i64 %dst2_size, i1 false)
  ret void
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1)
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1)
declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i1)
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture readonly, i32, i1)
