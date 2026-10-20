; RUN: llc -global-isel -global-isel-abort=2 -stop-after=aarch64-prelegalizer-combiner -verify-machineinstrs -o - < %s | FileCheck %s

target datalayout = "e-m:w-p:64:64-i32:32-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-windows-msvc"

define void @copy_to_ptr32(ptr addrspace(271) %dst, ptr %src) {
; CHECK-LABEL: name: copy_to_ptr32
; CHECK:       %[[SRC_OFFSET:[0-9]+]]:_(i64) = G_CONSTANT i64 4
; CHECK:       G_PTR_ADD %{{[0-9]+}}, %[[SRC_OFFSET]](i64)
; CHECK:       %[[DST_OFFSET:[0-9]+]]:_(i32) = G_CONSTANT i32 4
; CHECK:       G_PTR_ADD %{{[0-9]+}}, %[[DST_OFFSET]](i32)
  call void @llvm.memcpy.p271.p0.i64(ptr addrspace(271) align 2 %dst,
                                     ptr align 2 %src, i64 6, i1 false)
  ret void
}

define void @copy_from_ptr32(ptr %dst, ptr addrspace(271) %src) {
; CHECK-LABEL: name: copy_from_ptr32
; CHECK:       %[[SRC_OFFSET:[0-9]+]]:_(i32) = G_CONSTANT i32 4
; CHECK:       G_PTR_ADD %{{[0-9]+}}, %[[SRC_OFFSET]](i32)
; CHECK:       %[[DST_OFFSET:[0-9]+]]:_(i64) = G_CONSTANT i64 4
; CHECK:       G_PTR_ADD %{{[0-9]+}}, %[[DST_OFFSET]](i64)
  call void @llvm.memcpy.p0.p271.i64(ptr align 2 %dst,
                                     ptr addrspace(271) align 2 %src,
                                     i64 6, i1 false)
  ret void
}

declare void @llvm.memcpy.p271.p0.i64(ptr addrspace(271), ptr, i64, i1 immarg)
declare void @llvm.memcpy.p0.p271.i64(ptr, ptr addrspace(271), i64, i1 immarg)