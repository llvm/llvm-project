; RUN: opt -basicaa -loop-distribute -enable-loop-distribute -S < %s | \
; RUN:     FileCheck %s --check-prefix=ALL --check-prefix=STRIDE_SPEC

; RUN: opt -basicaa -loop-distribute -enable-loop-distribute -S -enable-mem-access-versioning=0 < %s | \
; RUN:     FileCheck %s --check-prefix=ALL --check-prefix=NO_STRIDE_SPEC

; If we don't speculate stride for 1 we can't distribute along the line
; because we could have a backward dependence:
;
;   for (i = 0; i < n; i++) {
;     A[i + 1] = A[i] * B[i];
;     =======================
;     C[i] = D[i] * A[stride * i];
;   }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

; ALL-LABEL: @f(
define void @f(i32* noalias %a,
               i32* noalias %b,
               i32* noalias %c,
               i32* noalias %d,
               i64 %stride) {
entry:
  br label %for.body

; STRIDE_SPEC: %ident.check = icmp ne i64 %stride, 1

; STRIDE_SPEC: for.body.ldist1:
; NO_STRIDE_SPEC-NOT: for.body.ldist1:

for.body:                                         ; preds = %for.body, %entry
  %ind = phi i64 [ 0, %entry ], [ %add, %for.body ]

  %arrayidxA = getelementptr inbounds i32, i32* %a, i64 %ind
  %loadA = load i32, i32* %arrayidxA, align 4

  %arrayidxB = getelementptr inbounds i32, i32* %b, i64 %ind
  %loadB = load i32, i32* %arrayidxB, align 4

  %mulA = mul i32 %loadB, %loadA

  %add = add nuw nsw i64 %ind, 1
  %arrayidxA_plus_4 = getelementptr inbounds i32, i32* %a, i64 %add
  store i32 %mulA, i32* %arrayidxA_plus_4, align 4

  %arrayidxD = getelementptr inbounds i32, i32* %d, i64 %ind
  %loadD = load i32, i32* %arrayidxD, align 4

  %mul = mul i64 %ind, %stride
  %arrayidxStridedA = getelementptr inbounds i32, i32* %a, i64 %mul
  %loadStridedA = load i32, i32* %arrayidxStridedA, align 4

  %mulC = mul i32 %loadD, %loadStridedA

  %arrayidxC = getelementptr inbounds i32, i32* %c, i64 %ind
  store i32 %mulC, i32* %arrayidxC, align 4

  %exitcond = icmp eq i64 %add, 20
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
