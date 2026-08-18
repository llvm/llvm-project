; RUN: opt < %s -passes=infer-alignment -S | FileCheck %s

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1)
declare void @llvm.memmove.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1)
declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1)

define void @memcpy(i64 %len) {
; CHECK-LABEL: define void @memcpy(
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 32 %dst, ptr align 16 %src, i64 %len, i1 false)
  %dst = alloca [64 x i8], align 32
  %src = alloca [64 x i8], align 16
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 %len, i1 false)
  ret void
}

define void @memmove(i64 %len) {
; CHECK-LABEL: define void @memmove(
; CHECK: call void @llvm.memmove.p0.p0.i64(ptr align 64 %dst, ptr align 8 %src, i64 %len, i1 false)
  %dst = alloca [64 x i8], align 64
  %src = alloca [64 x i8], align 8
  call void @llvm.memmove.p0.p0.i64(ptr %dst, ptr %src, i64 %len, i1 false)
  ret void
}

define void @memset(i64 %len) {
; CHECK-LABEL: define void @memset(
; CHECK: call void @llvm.memset.p0.i64(ptr align 32 %dst, i8 0, i64 %len, i1 false)
  %dst = alloca [64 x i8], align 32
  call void @llvm.memset.p0.i64(ptr %dst, i8 0, i64 %len, i1 false)
  ret void
}
