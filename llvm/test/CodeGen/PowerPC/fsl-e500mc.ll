;
; Test support for Freescale e500mc and its higher memcpy inlining thresholds.
;
; RUN: llc -verify-machineinstrs -mcpu=e500mc < %s 2>&1 | FileCheck %s
; CHECK-NOT: not a recognized processor for this target

target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32"
target triple = "powerpc-fsl-linux"

%struct.teststruct = type { [12 x i32], i32 }

define void @copy(%struct.teststruct* noalias nocapture sret %agg.result, %struct.teststruct* nocapture %in) nounwind {
entry:
; CHECK: @copy
; CHECK-NOT: bl memcpy
  %0 = bitcast %struct.teststruct* %agg.result to i8*
  %1 = bitcast %struct.teststruct* %in to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* %0, i8* %1, i32 52, i32 4, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
