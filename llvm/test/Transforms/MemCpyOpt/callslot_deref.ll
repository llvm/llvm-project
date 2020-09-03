; RUN: opt < %s -S -basic-aa -memcpyopt | FileCheck %s
target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1) unnamed_addr nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1) nounwind

; all bytes of %dst that are touch by the memset are dereferenceable
define void @must_remove_memcpy(i8* noalias nocapture dereferenceable(4096) %dst) {
; CHECK-LABEL: @must_remove_memcpy(
; CHECK: call void @llvm.memset.p0i8.i64
; CHECK-NOT: call void @llvm.memcpy.p0i8.p0i8.i64
  %src = alloca [4096 x i8], align 1
  %p = getelementptr inbounds [4096 x i8], [4096 x i8]* %src, i64 0, i64 0
  call void @llvm.memset.p0i8.i64(i8* %p, i8 0, i64 4096, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %p, i64 4096, i1 false) #2
  ret void
}

; memset touch more bytes than those guaranteed to be dereferenceable
; We can't remove the memcpy, but we can turn it into an independent memset.
define void @must_not_remove_memcpy(i8* noalias nocapture dereferenceable(1024) %dst) {
; CHECK-LABEL: @must_not_remove_memcpy(
; CHECK: call void @llvm.memset.p0i8.i64
; CHECK: call void @llvm.memset.p0i8.i64
  %src = alloca [4096 x i8], align 1
  %p = getelementptr inbounds [4096 x i8], [4096 x i8]* %src, i64 0, i64 0
  call void @llvm.memset.p0i8.i64(i8* %p, i8 0, i64 4096, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %p, i64 4096, i1 false) #2
  ret void
}
