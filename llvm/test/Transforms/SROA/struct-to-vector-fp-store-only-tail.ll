; RUN: opt -passes=sroa -S %s | FileCheck %s
; NOTE: Do not autogenerate. This regression test checks the direct SROA shape
; for a store-only homogeneous float slice under the simplified fallback rule.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%class.aiMatrix4x4t = type { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float }

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg)

; This reduced case exercises a homogeneous FP tail slice that is only stored.
; With the simplified fallback rule, SROA canonicalizes the first 16 bytes to a
; vector store and leaves the remaining 44 bytes as memcpy traffic.
;
; CHECK-LABEL: define ptr @store_only_fp_tail()
; CHECK: %.sroa.4 = alloca { float, float, float, float, float, float, float, float, float, float, float }, align 8
; CHECK: %.sroa.2 = alloca { float, float, float, float, float, float, float, float, float, float, float }, align 8
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %.sroa.4, ptr align 8 %.sroa.2, i64 44, i1 false)
; CHECK: %.sroa.01.0.vec.insert = insertelement <4 x float> {{.*}}, float 0.000000e+00, i32 0
; CHECK: store <4 x float> %.sroa.01.0.vec.insert, ptr null, align 1
; CHECK: store float 0.000000e+00, ptr getelementptr inbounds (i8, ptr null, i64 16), align 1
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 getelementptr inbounds (i8, ptr null, i64 20), ptr align 8 %.sroa.4, i64 44, i1 false)
define ptr @store_only_fp_tail() {
  %1 = alloca %class.aiMatrix4x4t, align 4
  %2 = alloca %class.aiMatrix4x4t, align 4
  %3 = getelementptr i8, ptr %2, i64 16
  store float 0.000000e+00, ptr %3, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr %1, ptr %2, i64 64, i1 false)
  store float 0.000000e+00, ptr %1, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr null, ptr %1, i64 64, i1 false)
  ret ptr null
}
