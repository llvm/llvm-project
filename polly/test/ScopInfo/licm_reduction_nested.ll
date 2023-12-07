; RUN: opt %loadPolly -basic-aa -loop-rotate -indvars       -polly-prepare -polly-print-scops -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -basic-aa -loop-rotate -indvars -licm -polly-prepare -polly-scops -disable-output < %s | FileCheck %s
;
; XFAIL: *
;
; Even ScopDetection fails here after LICM because of PHI in exit node.
;
;    void foo(unsigned long *restrict A, unsigned long *restrict B,
;             unsigned long j) {
;      for (unsigned long i = 0; i < 100; i++)
;        for (unsigned long k = 0; k < 100; k++)
;          A[j] += B[i + k];
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(ptr noalias %A, ptr noalias %B, i64 %j) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc.6, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc7, %for.inc.6 ]
  %exitcond1 = icmp ne i64 %i.0, 100
  br i1 %exitcond1, label %for.body, label %for.end.8

for.body:                                         ; preds = %for.cond
  br label %for.cond.1

for.cond.1:                                       ; preds = %for.inc, %for.body
  %k.0 = phi i64 [ 0, %for.body ], [ %inc, %for.inc ]
  %exitcond = icmp ne i64 %k.0, 100
  br i1 %exitcond, label %for.body.3, label %for.end

for.body.3:                                       ; preds = %for.cond.1
  %add = add nuw nsw i64 %i.0, %k.0
  %arrayidx = getelementptr inbounds i64, ptr %B, i64 %add
  %tmp = load i64, ptr %arrayidx, align 8
  %arrayidx4 = getelementptr inbounds i64, ptr %A, i64 %j
  %tmp2 = load i64, ptr %arrayidx4, align 8
  %add5 = add i64 %tmp2, %tmp
  store i64 %add5, ptr %arrayidx4, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body.3
  %inc = add nuw nsw i64 %k.0, 1
  br label %for.cond.1

for.end:                                          ; preds = %for.cond.1
  br label %for.inc.6

for.inc.6:                                        ; preds = %for.end
  %inc7 = add nuw nsw i64 %i.0, 1
  br label %for.cond

for.end.8:                                        ; preds = %for.cond
  ret void
}


; CHECK: Statements {
; CHECK:     Stmt_for_body_3
; CHECK-DAG:    ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:       [j] -> { Stmt_for_body_3[i0, i1] -> MemRef_B[i0 + i1] };
; CHECK-DAG:    ReadAccess :=       [Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:       [j] -> { Stmt_for_body_3[i0, i1] -> MemRef_A[j] };
; CHECK-DAG:    MustWriteAccess :=  [Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:       [j] -> { Stmt_for_body_3[i0, i1] -> MemRef_A[j] };
; CHECK: }
