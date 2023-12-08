; RUN: opt %loadPolly -polly-print-scops -disable-output \
; RUN: -polly-invariant-load-hoisting=true < %s | FileCheck %s
;
; CHECK:      Invariant Accesses: {
; CHECK-NEXT:       ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:         MemRef_bounds[0]
; CHECK-NEXT: Execution Context: [bounds0, bounds1] -> {  :  }
; CHECK-NEXT:       ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:         MemRef_bounds[1]
; CHECK-NEXT: Execution Context: [bounds0, bounds1] -> {  :  }
; CHECK:      }

;    double A[1000][1000];
;    long bounds[2];
;
;    void foo() {
;
;      for (long i = 0; i <= bounds[0]; i++)
;        for (long j = 0; j <= bounds[1]; j++)
;          A[i][j] += i + j;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@bounds = common global [2 x i64] zeroinitializer, align 16
@A = common global [1000 x [1000 x double]] zeroinitializer, align 16

define void @foo() {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc.6, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc7, %for.inc.6 ]
  %bounds0 = load i64, ptr @bounds, align 16
  %cmp = icmp sgt i64 %i.0, %bounds0
  br i1 %cmp, label %for.end.8, label %for.body

for.body:                                         ; preds = %for.cond
  br label %for.cond.1

for.cond.1:                                       ; preds = %for.inc, %for.body
  %j.0 = phi i64 [ 0, %for.body ], [ %inc, %for.inc ]
  %bounds1 = load i64, ptr getelementptr inbounds ([2 x i64], ptr @bounds, i64 0, i64 1), align 8
  %cmp2 = icmp sgt i64 %j.0, %bounds1
  br i1 %cmp2, label %for.end, label %for.body.3

for.body.3:                                       ; preds = %for.cond.1
  %add = add nsw i64 %i.0, %j.0
  %conv = sitofp i64 %add to double
  %arrayidx4 = getelementptr inbounds [1000 x [1000 x double]], ptr @A, i64 0, i64 %i.0, i64 %j.0
  %tmp2 = load double, ptr %arrayidx4, align 8
  %add5 = fadd double %tmp2, %conv
  store double %add5, ptr %arrayidx4, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body.3
  %inc = add nuw nsw i64 %j.0, 1
  br label %for.cond.1

for.end:                                          ; preds = %for.cond.1
  br label %for.inc.6

for.inc.6:                                        ; preds = %for.end
  %inc7 = add nuw nsw i64 %i.0, 1
  br label %for.cond

for.end.8:                                        ; preds = %for.cond
  ret void
}
