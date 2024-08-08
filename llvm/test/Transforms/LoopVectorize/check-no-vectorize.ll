; This test asserts that we don't emit both
; successful and unsuccessful message about vectorization.

; RUN: opt -passes=loop-vectorize -debug -disable-output -pass-remarks-missed=loop-vectorize %s 2>&1 | FileCheck %s
; CHECK-NOT: LV: We can vectorize this loop
; CHECK: LV: Not vectorizing: Cannot prove legality

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@a = dso_local global [32000 x i32] zeroinitializer, align 4
@b = dso_local global [32000 x i32] zeroinitializer, align 4

define dso_local void @foo() local_unnamed_addr {
entry:
  %.pre = load i32, ptr getelementptr inbounds (i8, ptr @a, i64 4), align 4
  %.pre17 = load i32, ptr @a, align 4
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %0 = phi i32 [ %.pre17, %entry ], [ %add6, %for.body ]
  %1 = phi i32 [ %.pre, %entry ], [ %2, %for.body ]
  %indvars.iv = phi i64 [ 1, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds [32000 x i32], ptr @a, i64 0, i64 %indvars.iv
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx2 = getelementptr inbounds [32000 x i32], ptr @a, i64 0, i64 %indvars.iv.next
  %2 = load i32, ptr %arrayidx2, align 4
  %add3 = add nsw i32 %2, %1
  %add6 = add nsw i32 %add3, %0
  store i32 %add6, ptr %arrayidx, align 4
  %exitcond.not = icmp eq i64 %indvars.iv.next, 31999
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
