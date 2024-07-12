; RUN: opt -passes='print<access-info>' -disable-output  < %s 2>&1 | FileCheck %s

; This loop:
;
;   int **A;
;   for (i)
;     for (j) {
;        A[i][j] = A[i-1][j] * B[j]
;        B[j+1] = 2       // backward dep between this and the previous
;     }
;
; is transformed by Load-PRE to stash away A[i] for the next iteration of the
; outer loop:
;
;   Curr = A[0];          // Prev_0
;   for (i: 1..N) {
;     Prev = Curr;        // Prev = PHI (Prev_0, Curr)
;     Curr = A[i];
;     for (j: 0..N) {
;        Curr[j] = Prev[j] * B[j]
;        B[j+1] = 2       // backward dep between this and the previous
;     }
;   }
;
; Since A[i] and A[i-1] are likely to be independent, getUnderlyingObjects
; should not assume that Curr and Prev share the same underlying object.
;
; If it did we would try to dependence-analyze Curr and Prev and the analysis
; would fail with non-constant distance.
;
; To illustrate one of the negative consequences of this, if the loop has a
; backward dependence we won't detect this but instead fully fall back on
; memchecks (that is what LAA does after encountering a case of non-constant
; distance).

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

; CHECK-LABEL: function 'f'
; CHECK: for_j.body:
; CHECK-NEXT: Report: unsafe dependent memory operations in loop
; CHECK-NEXT: Backward loop carried data dependence.
; CHECK-NEXT: Dependences:
; CHECK-NEXT: Backward:
; CHECK-NEXT: %loadB = load i8, ptr %gepB, align 1 ->
; CHECK-NEXT: store i8 2, ptr %gepB_plus_one, align 1

define void @f(ptr noalias %A, ptr noalias %B, i64 %N) {
for_i.preheader:
  %prev_0 = load ptr, ptr %A, align 8
  br label %for_i.body

for_i.body:
  %i = phi i64 [1, %for_i.preheader], [%i.1, %for_j.end]
  %prev = phi ptr [%prev_0, %for_i.preheader], [%curr, %for_j.end]
  %gep = getelementptr inbounds ptr, ptr %A, i64 %i
  %curr = load ptr, ptr %gep, align 8
  br label %for_j.preheader

for_j.preheader:
  br label %for_j.body

for_j.body:
  %j = phi i64 [0, %for_j.preheader], [%j.1, %for_j.body]

  %gepPrev = getelementptr inbounds i8, ptr %prev, i64 %j
  %gepCurr = getelementptr inbounds i8, ptr %curr, i64 %j
  %gepB = getelementptr inbounds i8, ptr %B, i64 %j

  %loadPrev = load i8, ptr %gepPrev, align 1
  %loadB = load i8, ptr %gepB, align 1

  %mul = mul i8 %loadPrev, %loadB

  store i8 %mul, ptr %gepCurr, align 1

  %gepB_plus_one = getelementptr inbounds i8, ptr %gepB, i64 1
  store i8 2, ptr %gepB_plus_one, align 1

  %j.1 = add nuw i64 %j, 1
  %exitcondj = icmp eq i64 %j.1, %N
  br i1 %exitcondj, label %for_j.end, label %for_j.body

for_j.end:

  %i.1 = add nuw i64 %i, 1
  %exitcond = icmp eq i64 %i.1, %N
  br i1 %exitcond, label %for_i.end, label %for_i.body

for_i.end:
  ret void
}

; CHECK-LABEL: function 'f_deep'
; CHECK: for_j.body:
; FIXME: This is incorrect and is going to be fixed with D86669.
; CHECK-NEXT: Memory dependences are safe with run-time checks
; CHECK-NEXT: Dependences:

define void @f_deep(ptr noalias %A, ptr noalias %B, i64 %N) {
for_i.preheader:
  %prev_0 = load ptr, ptr %A, align 8
  br label %for_i.body

for_i.body:
  %i = phi i64 [1, %for_i.preheader], [%i.1, %for_j.end]
  %prev = phi ptr [%prev_0, %for_i.preheader], [%curr, %for_j.end]
  %gep = getelementptr inbounds ptr, ptr %A, i64 %i
  %curr = load ptr, ptr %gep, align 8
  br label %for_j.preheader

for_j.preheader:
  br label %for_j.body

for_j.body:
  %j = phi i64 [0, %for_j.preheader], [%j.1, %for_j.body]

  %gepPrev = getelementptr inbounds i8, ptr %prev, i64 %j
  %gepCurr = getelementptr inbounds i8, ptr %curr, i64 %j
  %gepB = getelementptr inbounds i8, ptr %B, i64 %j
  %gepB1 = getelementptr inbounds i8, ptr %gepB, i64 %j
  %gepB2 = getelementptr inbounds i8, ptr %gepB1, i64 0
  %gepB3 = getelementptr inbounds i8, ptr %gepB2, i64 0
  %gepB4 = getelementptr inbounds i8, ptr %gepB3, i64 0
  %gepB5 = getelementptr inbounds i8, ptr %gepB4, i64 0
  %gepB6 = getelementptr inbounds i8, ptr %gepB5, i64 0
  %gepB7 = getelementptr inbounds i8, ptr %gepB6, i64 0
  %gepB8 = getelementptr inbounds i8, ptr %gepB7, i64 0
  %gepB9 = getelementptr inbounds i8, ptr %gepB8, i64 0

  %loadPrev = load i8, ptr %gepPrev, align 1
  %loadB = load i8, ptr %gepB9, align 1

  %mul = mul i8 %loadPrev, %loadB

  store i8 %mul, ptr %gepCurr, align 1

  %gepB_plus_one = getelementptr inbounds i8, ptr %gepB, i64 1
  store i8 2, ptr %gepB_plus_one, align 1

  %j.1 = add nuw i64 %j, 1
  %exitcondj = icmp eq i64 %j.1, %N
  br i1 %exitcondj, label %for_j.end, label %for_j.body

for_j.end:

  %i.1 = add nuw i64 %i, 1
  %exitcond = icmp eq i64 %i.1, %N
  br i1 %exitcond, label %for_i.end, label %for_i.body

for_i.end:
  ret void
}
