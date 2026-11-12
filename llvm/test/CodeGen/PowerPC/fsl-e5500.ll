;
; Test support for Freescale e5500 and its higher memcpy inlining thresholds.
;
; RUN: llc -verify-machineinstrs -mcpu=e5500 < %s 2>&1 | FileCheck %s
; CHECK-NOT: not a recognized processor for this target

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-fsl-linux"

%struct.teststruct = type { [24 x i32], i32 }

define void @copy(ptr noalias nocapture sret(%struct.teststruct) %agg.result, ptr nocapture %in) nounwind {
entry:
; CHECK: @copy
; CHECK-NOT: bl memcpy
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.result, ptr align 4 %in, i64 100, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1) nounwind
