; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -polly-codegen -analyze < %s
;
; void or(float *A, long n, long m) {
;   for (long i = 0; i < 100; i++) {
;     if (i < n || i < m)
;       A[i] += i;
;   }
; }
;
; void and(float *A, long n, long m) {
;   for (long i = 0; i < 100; i++) {
;     if (i < n && i < m)
;       A[i] += i;
;   }
; }

; CHECK-LABEL: Function: or
;
; CHECK:       Statements {
; CHECK-NEXT:      Stmt_if_then
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [n, m] -> { Stmt_if_then[i0] : (i0 <= 99 and i0 >= 0 and i0 <= -1 + m) or (i0 <= 99 and i0 >= 0 and i0 <= -1 + n) };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [n, m] -> { Stmt_if_then[i0] -> [i0] : i0 <= -1 + m or i0 <= -1 + n };
; CHECK-NEXT:          ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:              [n, m] -> { Stmt_if_then[i0] -> MemRef_A[i0] };
; CHECK-NEXT:          MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:              [n, m] -> { Stmt_if_then[i0] -> MemRef_A[i0] };
; CHECK-NEXT:  }
;
; CHECK-LABEL: Function: and
;
; CHECK:       Statements {
; CHECK-NEXT:      Stmt_if_then
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [n, m] -> { Stmt_if_then[i0] : i0 <= 99 and i0 >= 0 and i0 <= -1 + m and i0 <= -1 + n };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [n, m] -> { Stmt_if_then[i0] -> [i0] };
; CHECK-NEXT:          ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:              [n, m] -> { Stmt_if_then[i0] -> MemRef_A[i0] };
; CHECK-NEXT:          MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:              [n, m] -> { Stmt_if_then[i0] -> MemRef_A[i0] };
; CHECK-NEXT:  }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @or(float* nocapture %A, i64 %n, i64 %m) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %for.inc, %entry
  %i.03 = phi i64 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp1 = icmp slt i64 %i.03, %n
  %cmp2 = icmp slt i64 %i.03, %m
  %or.cond = or i1 %cmp1, %cmp2
  br i1 %or.cond, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %conv = sitofp i64 %i.03 to float
  %arrayidx = getelementptr inbounds float, float* %A, i64 %i.03
  %0 = load float, float* %arrayidx, align 4
  %add = fadd float %conv, %0
  store float %add, float* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %if.then, %for.body
  %inc = add nuw nsw i64 %i.03, 1
  %exitcond = icmp eq i64 %inc, 100
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc
  ret void
}

; Function Attrs: nounwind uwtable
define void @and(float* nocapture %A, i64 %n, i64 %m) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %for.inc, %entry
  %i.03 = phi i64 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp1 = icmp slt i64 %i.03, %n
  %cmp2 = icmp slt i64 %i.03, %m
  %or.cond = and i1 %cmp1, %cmp2
  br i1 %or.cond, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %conv = sitofp i64 %i.03 to float
  %arrayidx = getelementptr inbounds float, float* %A, i64 %i.03
  %0 = load float, float* %arrayidx, align 4
  %add = fadd float %conv, %0
  store float %add, float* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %inc = add nuw nsw i64 %i.03, 1
  %exitcond = icmp eq i64 %inc, 100
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc
  ret void
}
