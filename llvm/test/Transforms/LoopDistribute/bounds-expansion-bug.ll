; RUN: opt -basicaa -loop-distribute -S < %s | FileCheck %s

; When emitting the memchecks for:
;
;   for (i = 0; i < n; i++) {
;     A[i + 1] = A[i] * B[i];
;     =======================
;     C[i] = D[i] * E[i];
;   }
;
; we had a bug when expanding the bounds for A and C.  These are expanded
; multiple times and rely on the caching in SCEV expansion to avoid any
; redundancy.  However, due to logic in SCEVExpander::ReuseOrCreateCast, we
; can get earlier expanded values invalidated when casts are used.  This test
; ensure that we are not using the invalidated values.

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %a1, i32* %a2,
               i32* %b,
               i32* %c1, i32* %c2,
               i32* %d,
               i32* %e) {
entry:

  %cond = icmp eq i32* %e, null
  br i1 %cond, label %one, label %two
one:
  br label %join
two:
  br label %join
join:

; The pointers need to be defined by PHIs in order for the bug to trigger.
; Because of the PHIs the existing casts won't be at the desired location so a
; new cast will be emitted and the old cast will get invalidated.
;
; These are the steps:
;
; 1. After the bounds for A and C are first expanded:
;
;   join:
;     %a = phi i32* [ %a1, %one ], [ %a2, %two ]
;     %c = phi i32* [ %c1, %one ], [ %c2, %two ]
;     %c5 = bitcast i32* %c to i8*
;     %a3 = bitcast i32* %a to i8*
;
; 2. After A is expanded again:
;
;   join:                                             ; preds = %two, %one
;     %a = phi i32* [ %a1, %one ], [ %a2, %two ]
;     %c = phi i32* [ %c1, %one ], [ %c2, %two ]
;     %a3 = bitcast i32* %a to i8*                   <--- new
;     %c5 = bitcast i32* %c to i8*
;     %0 = bitcast i32* undef to i8*                 <--- old, invalidated
;
; 3. Finally, when C is expanded again:
;
;   join:                                             ; preds = %two, %one
;     %a = phi i32* [ %a1, %one ], [ %a2, %two ]
;     %c = phi i32* [ %c1, %one ], [ %c2, %two ]
;     %c5 = bitcast i32* %c to i8*                   <--- new
;     %a3 = bitcast i32* %a to i8*
;     %0 = bitcast i32* undef to i8*                 <--- old, invalidated
;     %1 = bitcast i32* undef to i8*

  %a = phi i32* [%a1, %one], [%a2, %two]
  %c = phi i32* [%c1, %one], [%c2, %two]
  br label %for.body


; CHECK: [[VALUE:%[0-9a-z]+]] = bitcast i32* undef to i8*
; CHECK-NOT: [[VALUE]]

for.body:                                         ; preds = %for.body, %entry
  %ind = phi i64 [ 0, %join ], [ %add, %for.body ]

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

  %arrayidxE = getelementptr inbounds i32, i32* %e, i64 %ind
  %loadE = load i32, i32* %arrayidxE, align 4

  %mulC = mul i32 %loadD, %loadE

  %arrayidxC = getelementptr inbounds i32, i32* %c, i64 %ind
  store i32 %mulC, i32* %arrayidxC, align 4

  %exitcond = icmp eq i64 %add, 20
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
