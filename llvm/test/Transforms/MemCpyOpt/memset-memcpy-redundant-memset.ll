; RUN: opt -basic-aa -memcpyopt -S %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; CHECK-LABEL: define void @test
; CHECK: [[ULE:%[0-9]+]] = icmp ule i64 %dst_size, %src_size
; CHECK: [[SIZEDIFF:%[0-9]+]] = sub i64 %dst_size, %src_size
; CHECK: [[SIZE:%[0-9]+]] = select i1 [[ULE]], i64 0, i64 [[SIZEDIFF]]
; CHECK: [[DST:%[0-9]+]] = getelementptr i8, i8* %dst, i64 %src_size
; CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* align 1 [[DST]], i8 %c, i64 [[SIZE]], i1 false)
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 %src_size, i1 false)
; CHECK-NEXT: ret void
define void @test(i8* %src, i64 %src_size, i8* %dst, i64 %dst_size, i8 %c) {
  call void @llvm.memset.p0i8.i64(i8* %dst, i8 %c, i64 %dst_size, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 %src_size, i1 false)
  ret void
}

; CHECK-LABEL: define void @test_different_types_i32_i64
; CHECK: [[DSTSIZE:%[0-9]+]] = zext i32 %dst_size to i64
; CHECK: [[ULE:%[0-9]+]] = icmp ule i64 [[DSTSIZE]], %src_size
; CHECK: [[SIZEDIFF:%[0-9]+]] = sub i64 [[DSTSIZE]], %src_size
; CHECK: [[SIZE:%[0-9]+]] = select i1 [[ULE]], i64 0, i64 [[SIZEDIFF]]
; CHECK: [[DST:%[0-9]+]] = getelementptr i8, i8* %dst, i64 %src_size
; CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* align 1 [[DST]], i8 %c, i64 [[SIZE]], i1 false)
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 %src_size, i1 false)
; CHECK-NEXT: ret void
define void @test_different_types_i32_i64(i8* %dst, i8* %src, i32 %dst_size, i64 %src_size, i8 %c) {
  call void @llvm.memset.p0i8.i32(i8* %dst, i8 %c, i32 %dst_size, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 %src_size, i1 false)
  ret void
}

; CHECK-LABEL: define void @test_different_types_i128_i32
; CHECK: [[SRCSIZE:%[0-9]+]] = zext i32 %src_size to i128
; CHECK: [[ULE:%[0-9]+]] = icmp ule i128 %dst_size, [[SRCSIZE]]
; CHECK: [[SIZEDIFF:%[0-9]+]] = sub i128 %dst_size, [[SRCSIZE]]
; CHECK: [[SIZE:%[0-9]+]] = select i1 [[ULE]], i128 0, i128 [[SIZEDIFF]]
; CHECK: [[DST:%[0-9]+]] = getelementptr i8, i8* %dst, i128 [[SRCSIZE]]
; CHECK-NEXT: call void @llvm.memset.p0i8.i128(i8* align 1 [[DST]], i8 %c, i128 [[SIZE]], i1 false)
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %src, i32 %src_size, i1 false)
; CHECK-NEXT: ret void
define void @test_different_types_i128_i32(i8* %dst, i8* %src, i128 %dst_size, i32 %src_size, i8 %c) {
  call void @llvm.memset.p0i8.i128(i8* %dst, i8 %c, i128 %dst_size, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %src, i32 %src_size, i1 false)
  ret void
}

; CHECK-LABEL: define void @test_different_types_i32_i128
; CHECK: [[DSTSIZE:%[0-9]+]] = zext i32 %dst_size to i128
; CHECK: [[ULE:%[0-9]+]] = icmp ule i128 [[DSTSIZE]], %src_size
; CHECK: [[SIZEDIFF:%[0-9]+]] = sub i128 [[DSTSIZE]], %src_size
; CHECK: [[SIZE:%[0-9]+]] = select i1 [[ULE]], i128 0, i128 [[SIZEDIFF]]
; CHECK: [[DST:%[0-9]+]] = getelementptr i8, i8* %dst, i128 %src_size
; CHECK-NEXT: call void @llvm.memset.p0i8.i128(i8* align 1 [[DST]], i8 %c, i128 [[SIZE]], i1 false)
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i128(i8* %dst, i8* %src, i128 %src_size, i1 false)
; CHECK-NEXT: ret void
define void @test_different_types_i32_i128(i8* %dst, i8* %src, i32 %dst_size, i128 %src_size, i8 %c) {
  call void @llvm.memset.p0i8.i32(i8* %dst, i8 %c, i32 %dst_size, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i128(i8* %dst, i8* %src, i128 %src_size, i1 false)
  ret void
}

; CHECK-LABEL: define void @test_different_types_i64_i32
; CHECK: [[SRCSIZE:%[0-9]+]] = zext i32 %src_size to i64
; CHECK: [[ULE:%[0-9]+]] = icmp ule i64 %dst_size, [[SRCSIZE]]
; CHECK: [[SIZEDIFF:%[0-9]+]] = sub i64 %dst_size, [[SRCSIZE]]
; CHECK: [[SIZE:%[0-9]+]] = select i1 [[ULE]], i64 0, i64 [[SIZEDIFF]]
; CHECK: [[DST:%[0-9]+]] = getelementptr i8, i8* %dst, i64 [[SRCSIZE]]
; CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* align 1 [[DST]], i8 %c, i64 [[SIZE]], i1 false)
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %src, i32 %src_size, i1 false)
; CHECK-NEXT: ret void
define void @test_different_types_i64_i32(i8* %dst, i8* %src, i64 %dst_size, i32 %src_size, i8 %c) {
  call void @llvm.memset.p0i8.i64(i8* %dst, i8 %c, i64 %dst_size, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %src, i32 %src_size, i1 false)
  ret void
}

; CHECK-LABEL: define void @test_align_same
; CHECK: call void @llvm.memset.p0i8.i64(i8* align 8 {{.*}}, i8 0, i64 {{.*}}, i1 false)
define void @test_align_same(i8* %src, i8* %dst, i64 %dst_size) {
  call void @llvm.memset.p0i8.i64(i8* align 8 %dst, i8 0, i64 %dst_size, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 80, i1 false)
  ret void
}

; CHECK-LABEL: define void @test_align_min
; CHECK: call void @llvm.memset.p0i8.i64(i8* align 4 {{.*}}, i8 0, i64 {{.*}}, i1 false)
define void @test_align_min(i8* %src, i8* %dst, i64 %dst_size) {
  call void @llvm.memset.p0i8.i64(i8* align 8 %dst, i8 0, i64 %dst_size, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 36, i1 false)
  ret void
}

; CHECK-LABEL: define void @test_align_memcpy
; CHECK: call void @llvm.memset.p0i8.i64(i8* align 8 {{.*}}, i8 0, i64 {{.*}}, i1 false)
define void @test_align_memcpy(i8* %src, i8* %dst, i64 %dst_size) {
  call void @llvm.memset.p0i8.i64(i8* %dst, i8 0, i64 %dst_size, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %dst, i8* align 8 %src, i64 80, i1 false)
  ret void
}

; CHECK-LABEL: define void @test_non_i8_dst_type
; CHECK-NEXT: %dst = bitcast i64* %dst_pi64 to i8*
; CHECK: [[ULE:%[0-9]+]] = icmp ule i64 %dst_size, %src_size
; CHECK: [[SIZEDIFF:%[0-9]+]] = sub i64 %dst_size, %src_size
; CHECK: [[SIZE:%[0-9]+]] = select i1 [[ULE]], i64 0, i64 [[SIZEDIFF]]
; CHECK: [[DST:%[0-9]+]] = getelementptr i8, i8* %dst, i64 %src_size
; CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* align 1 [[DST]], i8 %c, i64 [[SIZE]], i1 false)
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 %src_size, i1 false)
; CHECK-NEXT: ret void
define void @test_non_i8_dst_type(i8* %src, i64 %src_size, i64* %dst_pi64, i64 %dst_size, i8 %c) {
  %dst = bitcast i64* %dst_pi64 to i8*
  call void @llvm.memset.p0i8.i64(i8* %dst, i8 %c, i64 %dst_size, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 %src_size, i1 false)
  ret void
}

; CHECK-LABEL: define void @test_different_dst
; CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* %dst, i8 0, i64 %dst_size, i1 false)
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst2, i8* %src, i64 %src_size, i1 false)
; CHECK-NEXT: ret void
define void @test_different_dst(i8* %dst2, i8* %src, i64 %src_size, i8* %dst, i64 %dst_size) {
  call void @llvm.memset.p0i8.i64(i8* %dst, i8 0, i64 %dst_size, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst2, i8* %src, i64 %src_size, i1 false)
  ret void
}

; Make sure we also take into account dependencies on the destination.

; CHECK-LABEL: define i8 @test_intermediate_read
; CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* %a, i8 0, i64 64, i1 false)
; CHECK-NEXT: %r = load i8, i8* %a
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %a, i8* %b, i64 24, i1 false)
; CHECK-NEXT: ret i8 %r
define i8 @test_intermediate_read(i8* %a, i8* %b) #0 {
  call void @llvm.memset.p0i8.i64(i8* %a, i8 0, i64 64, i1 false)
  %r = load i8, i8* %a
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %a, i8* %b, i64 24, i1 false)
  ret i8 %r
}

%struct = type { [8 x i8], [8 x i8] }

; CHECK-LABEL: define void @test_intermediate_write
; CHECK-NEXT: %a = alloca %struct
; CHECK-NEXT: %a0 = getelementptr %struct, %struct* %a, i32 0, i32 0, i32 0
; CHECK-NEXT: %a1 = getelementptr %struct, %struct* %a, i32 0, i32 1, i32 0
; CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* %a0, i8 0, i64 16, i1 false)
; CHECK-NEXT: store i8 1, i8* %a1
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %a0, i8* %b, i64 8, i1 false)
; CHECK-NEXT: ret void
define void @test_intermediate_write(i8* %b) #0 {
  %a = alloca %struct
  %a0 = getelementptr %struct, %struct* %a, i32 0, i32 0, i32 0
  %a1 = getelementptr %struct, %struct* %a, i32 0, i32 1, i32 0
  call void @llvm.memset.p0i8.i64(i8* %a0, i8 0, i64 16, i1 false)
  store i8 1, i8* %a1
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %a0, i8* %b, i64 8, i1 false)
  ret void
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1)
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1)
declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i1)
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture readonly, i32, i1)
declare void @llvm.memset.p0i8.i128(i8* nocapture, i8, i128, i1)
declare void @llvm.memcpy.p0i8.p0i8.i128(i8* nocapture, i8* nocapture readonly, i128, i1)
