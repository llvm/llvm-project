; RUN: opt -basicaa -loop-vectorize -force-vector-width=2 \
; RUN:     -force-vector-interleave=1 -S < %s \
; RUN:     | FileCheck %s -check-prefix=BOTH -check-prefix=LV
; RUN: opt -basicaa -scoped-noalias -loop-vectorize -dse -force-vector-width=2 \
; RUN:     -force-vector-interleave=1 -S < %s \
; RUN:     | FileCheck %s -check-prefix=BOTH -check-prefix=DSE

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; This loop needs to be versioned with memchecks between {A, B} x {C} before
; it can be vectorized.
;
;   for (i = 0; i < n; i++) {
;     C[i] = A[i] + 1;
;     C[i] += B[i];
;   }
;
; Check that the corresponding noalias metadata is added to the vector loop
; but not to the scalar loop.
;
; Since in the versioned vector loop C and B can no longer alias, the first
; store to C[i] can be DSE'd.


define void @f(i32* %a, i32* %b, i32* %c) {
entry:
  br label %for.body

; BOTH: vector.memcheck:
; BOTH: vector.body:
for.body:                                         ; preds = %for.body, %entry
  %ind = phi i64 [ 0, %entry ], [ %inc, %for.body ]

  %arrayidxA = getelementptr inbounds i32, i32* %a, i64 %ind
; Scope 1
; LV: = load {{.*}} !alias.scope !0
  %loadA = load i32, i32* %arrayidxA, align 4

  %add = add nuw i32 %loadA, 2

  %arrayidxC = getelementptr inbounds i32, i32* %c, i64 %ind
; Noalias with scope 1 and 6
; LV: store {{.*}} !alias.scope !3, !noalias !5
; DSE-NOT: store
  store i32 %add, i32* %arrayidxC, align 4

  %arrayidxB = getelementptr inbounds i32, i32* %b, i64 %ind
; Scope 6
; LV: = load {{.*}} !alias.scope !7
  %loadB = load i32, i32* %arrayidxB, align 4

  %add2 = add nuw i32 %add, %loadB

; Noalias with scope 1 and 6
; LV: store {{.*}} !alias.scope !3, !noalias !5
; DSE: store
  store i32 %add2, i32* %arrayidxC, align 4

  %inc = add nuw nsw i64 %ind, 1
  %exitcond = icmp eq i64 %inc, 20
  br i1 %exitcond, label %for.end, label %for.body

; BOTH: for.body:
; BOTH-NOT: !alias.scope
; BOTH-NOT: !noalias

for.end:                                          ; preds = %for.body
  ret void
}

; LV: !0 = !{!1}
; LV: !1 = distinct !{!1, !2}
; LV: !2 = distinct !{!2, !"LVerDomain"}
; LV: !3 = !{!4}
; LV: !4 = distinct !{!4, !2}
; LV: !5 = !{!1, !6}
; LV: !6 = distinct !{!6, !2}
; LV: !7 = !{!6}
