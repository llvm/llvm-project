; RUN: opt -basicaa -loop-distribute -enable-loop-distribute -loop-simplify -scoped-noalias \
; RUN:     -loop-versioning -S < %s | FileCheck %s

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

@B = common global i32* null, align 8
@A = common global i32* null, align 8
@C = common global i32* null, align 8
@D = common global i32* null, align 8
@E = common global i32* null, align 8

define void @f() {
entry:
  %a = load i32*, i32** @A, align 8
  %b = load i32*, i32** @B, align 8
  %c = load i32*, i32** @C, align 8
  %d = load i32*, i32** @D, align 8
  %e = load i32*, i32** @E, align 8
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %ind = phi i64 [ 0, %entry ], [ %add, %for.body ]

  %arrayidxA = getelementptr inbounds i32, i32* %a, i64 %ind

; CHECK: %loadA.ldist1 = {{.*}} !noalias !25
; A noalias C: !25 -> { 17(15), 18(15), 19(15), 26(24) }
;                       ^^^^^^
  %loadA = load i32, i32* %arrayidxA, align 4

  %arrayidxB = getelementptr inbounds i32, i32* %b, i64 %ind
  %loadB = load i32, i32* %arrayidxB, align 4

  %mulA = mul i32 %loadB, %loadA

  %add = add nuw nsw i64 %ind, 1
  %arrayidxA_plus_4 = getelementptr inbounds i32, i32* %a, i64 %add
  store i32 %mulA, i32* %arrayidxA_plus_4, align 4

; CHECK: for.body:

  %arrayidxD = getelementptr inbounds i32, i32* %d, i64 %ind

; CHECK: %loadD = {{.*}} !alias.scope !31
; D's scope: !31 -> { 18(15), 32(33) }
;                             ^^^^^^
  %loadD = load i32, i32* %arrayidxD, align 4

  %arrayidxE = getelementptr inbounds i32, i32* %e, i64 %ind

; CHECK: %loadE = {{.*}} !alias.scope !34
; E's scope: !34 -> { 19(15), 35(33) }
;                             ^^^^^^
  %loadE = load i32, i32* %arrayidxE, align 4

  %mulC = mul i32 %loadD, %loadE

  %arrayidxC = getelementptr inbounds i32, i32* %c, i64 %ind

; CHECK: store i32 %mulC, {{.*}} !alias.scope !36, !noalias !38
; C's scope: !36 -> { 17(15), 37(33) }
;                     ^^^^^^
; C noalias D and E: !38 -> { 21(15), 32(33), 35(33) }
;                                     ^^^^^^  ^^^^^^
  store i32 %mulC, i32* %arrayidxC, align 4

  %exitcond = icmp eq i64 %add, 20
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; Domain for the second loop versioning for the top loop after
; distribution.
; CHECK: !15 = distinct !{!15, !"LVerDomain"}
; CHECK: !17 = distinct !{!17, !15}
; CHECK: !25 = !{!17, !18, !19, !26}
; CHECK: !31 = !{!18, !32}
; CHECK: !32 = distinct !{!32, !33}
; Domain for the second loop versioning for the bottom loop after
; distribution.
; CHECK: !33 = distinct !{!33, !"LVerDomain"}
; CHECK: !34 = !{!19, !35}
; CHECK: !35 = distinct !{!35, !33}
; CHECK: !36 = !{!17, !37}
; CHECK: !38 = !{!21, !32, !35}
