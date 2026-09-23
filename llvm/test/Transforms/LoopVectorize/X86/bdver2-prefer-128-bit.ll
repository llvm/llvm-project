; RUN: opt -mcpu=bdver2 -passes=loop-vectorize -S < %s | FileCheck %s --check-prefixes=CHECK,CHECK-PREFER-128
; RUN: opt -mcpu=bdver2 -mattr=-prefer-128-bit -passes=loop-vectorize -S < %s | FileCheck %s --check-prefixes=CHECK,CHECK-PREFER-256

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Bulldozer-family CPUs dual-pump 256-bit AVX ops as two 128-bit operations.
; Prefer 128-bit vectorization by default (see TuningPrefer128Bit on BdVer1Tuning).

; CHECK-LABEL: @test_loop(
; CHECK: vector.body:
; CHECK-PREFER-128: load <4 x float>
; CHECK-PREFER-128-NOT: load <8 x float>
; CHECK-PREFER-256: load <8 x float>

define void @test_loop(ptr noalias nocapture readonly %in, ptr noalias nocapture writeonly %out, i32 %n, float %scale, float %bias) {
entry:
  %cmp = icmp eq i32 %n, 0
  br i1 %cmp, label %exit, label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %idx = zext i32 %i to i64
  %in.ptr = getelementptr inbounds float, ptr %in, i64 %idx
  %val = load float, ptr %in.ptr, align 4
  %mul = fmul float %val, %scale
  %add = fadd float %mul, %bias
  %out.ptr = getelementptr inbounds float, ptr %out, i64 %idx
  store float %add, ptr %out.ptr, align 4
  %i.next = add nuw i32 %i, 1
  %cond = icmp eq i32 %i.next, %n
  br i1 %cond, label %exit, label %loop

exit:
  ret void
}
