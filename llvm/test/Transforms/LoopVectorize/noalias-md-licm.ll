; RUN: opt -basic-aa -scoped-noalias-aa -loop-vectorize -licm -force-vector-width=2 \
; RUN:     -force-vector-interleave=1 -S < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; In order to vectorize the inner loop, it needs to be versioned with
; memchecks between {A} x {B, C} first:
;
;   for (i = 0; i < n; i++)
;     for (j = 0; j < m; j++)
;         A[j] += B[i] + C[j];
;
; Since in the versioned vector loop A and B can no longer alias, B[i] can be
; LICM'ed from the inner loop.


define void @f(i32* %a, i32* %b, i32* %c) {
entry:
  br label %outer

outer:
  %i.2 = phi i64 [ 0, %entry ], [ %i, %inner.end ]
  %arrayidxB = getelementptr inbounds i32, i32* %b, i64 %i.2
  br label %inner.ph

inner.ph:
; CHECK: vector.ph:
; CHECK: load i32, i32* %arrayidxB,
; CHECK: br label %vector.body
  br label %inner

inner:
  %j.2 = phi i64 [ 0, %inner.ph ], [ %j, %inner ]

  %arrayidxA = getelementptr inbounds i32, i32* %a, i64 %j.2
  %loadA = load i32, i32* %arrayidxA, align 4

  %loadB = load i32, i32* %arrayidxB, align 4

  %arrayidxC = getelementptr inbounds i32, i32* %c, i64 %j.2
  %loadC = load i32, i32* %arrayidxC, align 4

  %add = add nuw i32 %loadA, %loadB
  %add2 = add nuw i32 %add, %loadC

  store i32 %add2, i32* %arrayidxA, align 4

  %j = add nuw nsw i64 %j.2, 1
  %cond1 = icmp eq i64 %j, 20
  br i1 %cond1, label %inner.end, label %inner

inner.end:
  %i = add nuw nsw i64 %i.2, 1
  %cond2 = icmp eq i64 %i, 30
  br i1 %cond2, label %outer.end, label %outer

outer.end:
  ret void
}
