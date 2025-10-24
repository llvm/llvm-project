; RUN: opt   -passes=loop-distribute,loop-simplify,loop-versioning -enable-loop-distribute -S < %s | FileCheck %s

; Test the metadata generated when versioning an already versioned loop.  Here
; we invoke loop distribution to perform the first round of versioning.  It
; adds memchecks for accesses that can alias across the distribution boundary.
; Then we further version the distributed loops to fully disambiguate accesses
; within each.
;
; So as an example, we add noalias between C and A during the versioning
; within loop distribution and then add noalias between C and D during the
; second explicit versioning step:
;
;   for (i = 0; i < n; i++) {
;     A[i + 1] = A[i] * B[i];
; -------------------------------
;     C[i] = D[i] * E[i];
;   }

; To see it easier what's going on, I expanded every noalias/scope metadata
; reference below in a comment.  For a scope I use the format scope(domain),
; e.g. scope 17 in domain 15 is written as 17(15).

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

@B = common global ptr null, align 8
@A = common global ptr null, align 8
@C = common global ptr null, align 8
@D = common global ptr null, align 8
@E = common global ptr null, align 8

define void @f() {
entry:
  %a = load ptr, ptr @A, align 8
  %b = load ptr, ptr @B, align 8
  %c = load ptr, ptr @C, align 8
  %d = load ptr, ptr @D, align 8
  %e = load ptr, ptr @E, align 8
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %ind = phi i64 [ 0, %entry ], [ %add, %for.body ]

  %arrayidxA = getelementptr inbounds i32, ptr %a, i64 %ind

; CHECK: %loadA.ldist1 = {{.*}} !alias.scope !24, !noalias !27
; A noalias C: !33 -> { 19(17), 20(17), 21(17), 28(26) }
;                       ^^^^^^
  %loadA = load i32, ptr %arrayidxA, align 4

  %arrayidxB = getelementptr inbounds i32, ptr %b, i64 %ind
  %loadB = load i32, ptr %arrayidxB, align 4

  %mulA = mul i32 %loadB, %loadA

  %add = add nuw nsw i64 %ind, 1
  %arrayidxA_plus_4 = getelementptr inbounds i32, ptr %a, i64 %add
  store i32 %mulA, ptr %arrayidxA_plus_4, align 4

; CHECK: for.body:

  %arrayidxD = getelementptr inbounds i32, ptr %d, i64 %ind

; CHECK: %loadD = {{.*}} !alias.scope !33
; D's scope: !33 -> { 20(17), 34(35) }
;                             ^^^^^^
  %loadD = load i32, ptr %arrayidxD, align 4

  %arrayidxE = getelementptr inbounds i32, ptr %e, i64 %ind

; CHECK: %loadE = {{.*}} !alias.scope !36
; E's scope: !36 -> { 21(17), 37(33) }
;                             ^^^^^^
  %loadE = load i32, ptr %arrayidxE, align 4

  %mulC = mul i32 %loadD, %loadE

  %arrayidxC = getelementptr inbounds i32, ptr %c, i64 %ind

; CHECK: store i32 %mulC, {{.*}} !alias.scope !38, !noalias !40
; C's scope: !38 -> { 19(17), 39(35)
;                     ^^^^^^
; C noalias D and E: !38 -> { 21(15), 32(33), 35(33) }
;                                     ^^^^^^  ^^^^^^
  store i32 %mulC, ptr %arrayidxC, align 4

  %exitcond = icmp eq i64 %add, 20
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; Domain for the second loop versioning for the top loop after
; distribution.
; CHECK: !17 = distinct !{!17, !"LVerDomain"}
; CHECK: !19 = distinct !{!19, !17}
; CHECK: !20 = distinct !{!20, !17}
; CHECK: !21 = distinct !{!21, !17}
; CHECK: !27 = !{!19, !20, !21, !28}
; CHECK: !28 = distinct !{!28, !26}
; CHECK: !33 = !{!20, !34}
; CHECK: !34 = distinct !{!34, !35}
; Domain for the second loop versioning for the bottom loop after
; distribution.
; CHECK: !35 = distinct !{!35, !"LVerDomain"}
; CHECK: !36 = !{!21, !37}
; CHECK: !37 = distinct !{!37, !35}
; CHECK: !38 = !{!19, !39}
; CHECK: !39 = distinct !{!39, !35}
; CHECK: !40 = !{!23, !34, !37}
