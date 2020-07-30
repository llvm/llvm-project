; RUN: opt %loadPolly -basic-aa -loop-rotate -indvars       -polly-prepare -polly-scops -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -basic-aa -loop-rotate -indvars -licm -polly-prepare -polly-scops -analyze < %s | FileCheck %s
;
; XFAIL: *
;
;    void foo(float *restrict A, float *restrict B, long j) {
;      for (long i = 0; i < 100; i++)
;        A[j] = B[i];
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(float* noalias %A, float* noalias %B, i64 %j) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i64 %i.0, 100
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds float, float* %B, i64 %i.0
  %tmp = bitcast float* %arrayidx to i32*
  %tmp1 = load i32, i32* %tmp, align 4
  %arrayidx1 = getelementptr inbounds float, float* %A, i64 %j
  %tmp2 = bitcast float* %arrayidx1 to i32*
  store i32 %tmp1, i32* %tmp2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nuw nsw i64 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; CHECK: Statements {
; CHECK:     Stmt_for_body
; CHECK-DAG:    ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:       [j] -> { Stmt_for_body[i0] -> MemRef_B[i0] };
; CHECK-DAG:    MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:       [j] -> { Stmt_for_body[i0] -> MemRef_A[j] };
; CHECK: }
