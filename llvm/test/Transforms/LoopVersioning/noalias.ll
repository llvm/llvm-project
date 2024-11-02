; RUN: opt -passes=loop-versioning -S < %s | FileCheck %s

; A very simple case.  After versioning the %loadA and %loadB can't alias with
; the store.
;
; To see it easier what's going on, I expanded every noalias/scope metadata
; reference below in a comment.  For a scope I use the format scope(domain),
; e.g. scope 17 in domain 15 is written as 17(15).

; CHECK-LABEL: @f(

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %a, i32* %b, i32* %c) {
entry:
  br label %for.body

; CHECK: for.body.lver.orig:
; CHECK: for.body:
for.body:                                         ; preds = %for.body, %entry
  %ind = phi i64 [ 0, %entry ], [ %add, %for.body ]

  %arrayidxA = getelementptr inbounds i32, i32* %a, i64 %ind
; CHECK: %loadA = {{.*}} !alias.scope !0
; A's scope: !0 -> { 1(2) }
  %loadA = load i32, i32* %arrayidxA, align 4

  %arrayidxB = getelementptr inbounds i32, i32* %b, i64 %ind
; CHECK: %loadB = {{.*}} !alias.scope !3
; B's scope: !3 -> { 4(2) }
  %loadB = load i32, i32* %arrayidxB, align 4

  %mulC = mul i32 %loadA, %loadB

  %arrayidxC = getelementptr inbounds i32, i32* %c, i64 %ind
; CHECK: store {{.*}} !alias.scope !5, !noalias !7
; C noalias A and B: !7 -> { 1(2), 4(2) }
  store i32 %mulC, i32* %arrayidxC, align 4

  %add = add nuw nsw i64 %ind, 1
  %exitcond = icmp eq i64 %add, 20
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
; CHECK: !0 = !{!1}
; CHECK: !1 = distinct !{!1, !2}
; CHECK: !2 = distinct !{!2, !"LVerDomain"}
; CHECK: !3 = !{!4}
; CHECK: !4 = distinct !{!4, !2}
; CHECK: !5 = !{!6}
; CHECK: !6 = distinct !{!6, !2}
; CHECK: !7 = !{!1, !4}
