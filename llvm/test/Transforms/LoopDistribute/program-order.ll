; RUN: opt -passes=loop-distribute -enable-loop-distribute -S -verify-loop-info -verify-dom-info < %s \
; RUN:   | FileCheck %s

; Distributing this loop to avoid the dependence cycle would require to
; reorder S1 and S2 to form the two partitions: {S2} | {S1, S3}.  The analysis
; provided by LoopAccessAnalysis does not allow us to reorder memory
; operations so make sure we bail on this loop.
;
;   for (i = 0; i < n; i++) {
;     S1: d = D[i];
;     S2: A[i + 1] = A[i] * B[i];
;     S3: C[i] = d * E[i];
;   }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

define void @f(ptr noalias %a,
               ptr noalias %b,
               ptr noalias %c,
               ptr noalias %d,
               ptr noalias %e) {
entry:
  br label %for.body

; CHECK: entry:
; CHECK:    br label %for.body
; CHECK: for.body:
; CHECK:    br i1 %exitcond, label %for.end, label %for.body
; CHECK: for.end:
; CHECK:    ret void

for.body:                                         ; preds = %for.body, %entry
  %ind = phi i64 [ 0, %entry ], [ %add, %for.body ]

  %arrayidxA = getelementptr inbounds i32, ptr %a, i64 %ind
  %loadA = load i32, ptr %arrayidxA, align 4

  %arrayidxB = getelementptr inbounds i32, ptr %b, i64 %ind
  %loadB = load i32, ptr %arrayidxB, align 4

  %mulA = mul i32 %loadB, %loadA

  %arrayidxD = getelementptr inbounds i32, ptr %d, i64 %ind
  %loadD = load i32, ptr %arrayidxD, align 4

  %add = add nuw nsw i64 %ind, 1
  %arrayidxA_plus_4 = getelementptr inbounds i32, ptr %a, i64 %add
  store i32 %mulA, ptr %arrayidxA_plus_4, align 4

  %arrayidxC = getelementptr inbounds i32, ptr %c, i64 %ind

  %arrayidxE = getelementptr inbounds i32, ptr %e, i64 %ind
  %loadE = load i32, ptr %arrayidxE, align 4

  %mulC = mul i32 %loadD, %loadE

  store i32 %mulC, ptr %arrayidxC, align 4

  %exitcond = icmp eq i64 %add, 20
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
