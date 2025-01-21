; RUN: opt -S < %s -passes=loop-vectorize -mcpu=generic -mattr=+sve2 | FileCheck %s --check-prefix=CHECK-GENERIC
; RUN: opt -S < %s -passes=loop-vectorize -mcpu=neoverse-v2 | FileCheck %s --check-prefix=CHECK-PREFERFIXED
; RUN: opt -S < %s -passes=loop-vectorize -mcpu=cortex-x2 | FileCheck %s --check-prefix=CHECK-PREFERFIXED
; RUN: opt -S < %s -passes=loop-vectorize -mcpu=cortex-x3 | FileCheck %s --check-prefix=CHECK-PREFERFIXED
; RUN: opt -S < %s -passes=loop-vectorize -mcpu=cortex-x4 | FileCheck %s --check-prefix=CHECK-PREFERFIXED
; RUN: opt -S < %s -passes=loop-vectorize -mcpu=cortex-x925 | FileCheck %s --check-prefix=CHECK-PREFERFIXED

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@a = dso_local local_unnamed_addr global [32000 x float] zeroinitializer, align 64
@b = dso_local local_unnamed_addr global [32000 x float] zeroinitializer, align 64

define void @test() #0 {
; CHECK-GENERIC-LABEL: define void @test(
; CHECK-GENERIC:       store <vscale x 4 x float>
; CHECK-PREFERFIXED-LABEL: define void @test(
; CHECK-PREFERFIXED:       store <4 x float>
;
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds [32000 x float], ptr @a, i64 0, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds [32000 x float], ptr @b, i64 0, i64 %indvars.iv
  %1 = load float, ptr %arrayidx2, align 4
  %add = fadd fast float %1, %0
  %2 = add nuw nsw i64 %indvars.iv, 16000
  %arrayidx5 = getelementptr inbounds [32000 x float], ptr @a, i64 0, i64 %2
  store float %add, ptr %arrayidx5, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 16000
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

attributes #0 = { vscale_range(1,16) }
