; RUN: opt -instcombine -unfold-element-atomic-memcpy-max-elements=8 -S < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Test basic unfolding
define void @test1(i8* %Src, i8* %Dst) {
; CHECK-LABEL: test1
; CHECK-NOT: llvm.memcpy.element.atomic

; CHECK-DAG: %memcpy_unfold.src_casted = bitcast i8* %Src to i32*
; CHECK-DAG: %memcpy_unfold.dst_casted = bitcast i8* %Dst to i32*

; CHECK-DAG: [[VAL1:%[^\s]+]] =  load atomic i32, i32* %memcpy_unfold.src_casted unordered, align 4
; CHECK-DAG: store atomic i32 [[VAL1]], i32* %memcpy_unfold.dst_casted unordered, align 8

; CHECK-DAG: [[VAL2:%[^\s]+]] =  load atomic i32, i32* %{{[^\s]+}} unordered, align 4
; CHECK-DAG: store atomic i32 [[VAL2]], i32* %{{[^\s]+}} unordered, align 4

; CHECK-DAG: [[VAL3:%[^\s]+]] =  load atomic i32, i32* %{{[^\s]+}} unordered, align 4
; CHECK-DAG: store atomic i32 [[VAL3]], i32* %{{[^\s]+}} unordered, align 4

; CHECK-DAG: [[VAL4:%[^\s]+]] =  load atomic i32, i32* %{{[^\s]+}} unordered, align 4
; CHECK-DAG: store atomic i32 [[VAL4]], i32* %{{[^\s]+}} unordered, align 4
entry:
  call void @llvm.memcpy.element.atomic.p0i8.p0i8(i8* align 4 %Dst, i8* align 8 %Src, i64 4, i32 4)
  ret void
}

; Test that we don't unfold too much
define void @test2(i8* %Src, i8* %Dst) {
; CHECK-LABEL: test2

; CHECK-NOT: load
; CHECK-NOT: store
; CHECK: llvm.memcpy.element.atomic
entry:
  call void @llvm.memcpy.element.atomic.p0i8.p0i8(i8* align 4 %Dst, i8* align 4 %Src, i64 1000, i32 4)
  ret void
}

; Test that we will not unfold into non native integers
define void @test3(i8* %Src, i8* %Dst) {
; CHECK-LABEL: test3

; CHECK-NOT: load
; CHECK-NOT: store
; CHECK: llvm.memcpy.element.atomic
entry:
  call void @llvm.memcpy.element.atomic.p0i8.p0i8(i8* align 64 %Dst, i8* align 64 %Src, i64 4, i32 64)
  ret void
}

; Test that we will eliminate redundant bitcasts
define void @test4(i64* %Src, i64* %Dst) {
; CHECK-LABEL: test4
; CHECK-NOT: llvm.memcpy.element.atomic

; CHECK-NOT: bitcast

; CHECK-DAG: [[VAL1:%[^\s]+]] =  load atomic i64, i64* %Src unordered, align 16
; CHECK-DAG: store atomic i64 [[VAL1]], i64* %Dst unordered, align 16

; CHECK-DAG: [[SRC_ADDR2:%[^ ]+]] = getelementptr i64, i64* %Src, i64 1
; CHECK-DAG: [[DST_ADDR2:%[^ ]+]] = getelementptr i64, i64* %Dst, i64 1
; CHECK-DAG: [[VAL2:%[^\s]+]] =  load atomic i64, i64* [[SRC_ADDR2]] unordered, align 8
; CHECK-DAG: store atomic i64 [[VAL2]], i64* [[DST_ADDR2]] unordered, align 8

; CHECK-DAG: [[SRC_ADDR3:%[^ ]+]] = getelementptr i64, i64* %Src, i64 2
; CHECK-DAG: [[DST_ADDR3:%[^ ]+]] = getelementptr i64, i64* %Dst, i64 2
; CHECK-DAG: [[VAL3:%[^ ]+]] =  load atomic i64, i64* [[SRC_ADDR3]] unordered, align 8
; CHECK-DAG: store atomic i64 [[VAL3]], i64* [[DST_ADDR3]] unordered, align 8

; CHECK-DAG: [[SRC_ADDR4:%[^ ]+]] = getelementptr i64, i64* %Src, i64 3
; CHECK-DAG: [[DST_ADDR4:%[^ ]+]] = getelementptr i64, i64* %Dst, i64 3
; CHECK-DAG: [[VAL4:%[^ ]+]] =  load atomic i64, i64* [[SRC_ADDR4]] unordered, align 8
; CHECK-DAG: store atomic i64 [[VAL4]], i64* [[DST_ADDR4]] unordered, align 8
entry:
  %Src.casted = bitcast i64* %Src to i8*
  %Dst.casted = bitcast i64* %Dst to i8*
  call void @llvm.memcpy.element.atomic.p0i8.p0i8(i8* align 16 %Dst.casted, i8* align 16 %Src.casted, i64 4, i32 8)
  ret void
}

define void @test5(i8* %Src, i8* %Dst) {
; CHECK-LABEL: test5

; CHECK-NOT: llvm.memcpy.element.atomic.p0i8.p0i8(i8* align 64 %Dst, i8* align 64 %Src, i64 0, i32 64)
entry:
  call void @llvm.memcpy.element.atomic.p0i8.p0i8(i8* align 64 %Dst, i8* align 64 %Src, i64 0, i32 64)
  ret void
}

declare void @llvm.memcpy.element.atomic.p0i8.p0i8(i8* nocapture, i8* nocapture, i64, i32)
