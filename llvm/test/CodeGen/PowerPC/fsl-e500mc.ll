;
; Test support for Freescale e500mc and its higher memcpy inlining thresholds.
;
; RUN: llc -verify-machineinstrs -mcpu=e500mc < %s 2>&1 | FileCheck %s
; CHECK-NOT: not a recognized processor for this target

target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32"
target triple = "powerpc-fsl-linux"

%struct.teststruct = type { [12 x i32], i32 }

define void @copy(ptr noalias nocapture sret(%struct.teststruct) %agg.result, ptr nocapture %in) nounwind {
entry:
; CHECK: @copy
; CHECK-NOT: bl memcpy
  tail call void @llvm.memcpy.p0.p0.i32(ptr align 4 %agg.result, ptr align 4 %in, i32 52, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1) nounwind
