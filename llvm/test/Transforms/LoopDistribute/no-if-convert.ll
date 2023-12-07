; RUN: opt -passes=loop-distribute -enable-loop-distribute -verify-loop-info -verify-dom-info -S < %s \
; RUN:   | FileCheck %s

; We should distribute this loop along === but not along ---.  The last
; partition won't be vectorized due to conditional stores so it's better to
; keep it with the second partition which has a dependence cycle.

; (1st statement):
;   for (i = 0; i < n; i++) {
;     C[i] = D[i] * E[i];
;=============================
;     A[i + 1] = A[i] * B[i];
;-----------------------------
;     if (F[i])
;        G[i] = H[i] * J[i];
;   }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

define void @f(ptr noalias %a,
               ptr noalias %b,
               ptr noalias %c,
               ptr noalias %d,
               ptr noalias %e,
               ptr noalias %g,
               ptr noalias %h,
               ptr noalias %j,
               i64 %x) {
entry:
  br label %for.body

; Ensure that we have only two partitions, the first with one multiplication
; and the second with two.

; CHECK: for.body.ldist1:
; CHECK:    %mulC.ldist1 = mul i32 %loadD.ldist1, %loadE.ldist1
; CHECK:    br i1 %exitcond.ldist1, label %entry.split, label %for.body.ldist1
; CHECK: entry.split:
; CHECK:    br label %for.body
; CHECK: for.body:
; CHECK:    %mulA = mul i32 %loadB, %loadA
; CHECK:    %mulG = mul i32 %loadH, %loadJ
; CHECK: for.end:

for.body:                                         ; preds = %for.body, %entry
  %ind = phi i64 [ 0, %entry ], [ %add, %if.end ]

  %arrayidxD = getelementptr inbounds i32, ptr %d, i64 %ind
  %loadD = load i32, ptr %arrayidxD, align 4

  %arrayidxE = getelementptr inbounds i32, ptr %e, i64 %ind
  %loadE = load i32, ptr %arrayidxE, align 4

  %mulC = mul i32 %loadD, %loadE

  %arrayidxC = getelementptr inbounds i32, ptr %c, i64 %ind
  store i32 %mulC, ptr %arrayidxC, align 4


  %arrayidxA = getelementptr inbounds i32, ptr %a, i64 %ind
  %loadA = load i32, ptr %arrayidxA, align 4

  %arrayidxB = getelementptr inbounds i32, ptr %b, i64 %ind
  %loadB = load i32, ptr %arrayidxB, align 4

  %mulA = mul i32 %loadB, %loadA

  %add = add nuw nsw i64 %ind, 1
  %arrayidxA_plus_4 = getelementptr inbounds i32, ptr %a, i64 %add
  store i32 %mulA, ptr %arrayidxA_plus_4, align 4

  %if.cond = icmp eq i64 %ind, %x
  br i1 %if.cond, label %if.then, label %if.end

if.then:
  %arrayidxH = getelementptr inbounds i32, ptr %h, i64 %ind
  %loadH = load i32, ptr %arrayidxH, align 4

  %arrayidxJ = getelementptr inbounds i32, ptr %j, i64 %ind
  %loadJ = load i32, ptr %arrayidxJ, align 4

  %mulG = mul i32 %loadH, %loadJ

  %arrayidxG = getelementptr inbounds i32, ptr %g, i64 %ind
  store i32 %mulG, ptr %arrayidxG, align 4
  br label %if.end

if.end:
  %exitcond = icmp eq i64 %add, 20
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
