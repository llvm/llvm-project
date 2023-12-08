; RUN: opt < %s -passes='print<loop-cache-cost>' -disable-output 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; void foo(long n, long m, long o, int A[n][m][o]) {
;   for (long i = 0; i < n; i++)
;     for (long j = 0; j < m; j++)
;       for (long k = 0; k < o; k++)
;         A[2*i+3][3*j-4][2*k+7] = 1;
; }

; CHECK: Loop 'for.i' has cost = 100000000
; CHECK-NEXT: Loop 'for.j' has cost = 1000000
; CHECK-NEXT: Loop 'for.k' has cost = 60000

define void @foo(i64 %n, i64 %m, i64 %o, i32* %A) {
entry:
  %cmp32 = icmp sgt i64 %n, 0
  %cmp230 = icmp sgt i64 %m, 0
  %cmp528 = icmp sgt i64 %o, 0
  br i1 %cmp32, label %for.cond1.preheader.lr.ph, label %for.end

for.cond1.preheader.lr.ph:                        ; preds = %entry
  br i1 %cmp230, label %for.i.preheader, label %for.end

for.i.preheader:                                  ; preds = %for.cond1.preheader.lr.ph
  br i1 %cmp528, label %for.i.preheader.split, label %for.end

for.i.preheader.split:                            ; preds = %for.i.preheader
  br label %for.i

for.i:                                            ; preds = %for.inci, %for.i.preheader.split
  %i = phi i64 [ %inci, %for.inci ], [ 0, %for.i.preheader.split ]
  %mul8 = shl i64 %i, 1
  %add9 = add nsw i64 %mul8, 3
  %0 = mul i64 %add9, %m
  %sub = add i64 %0, -4
  br label %for.j

for.j:                                            ; preds = %for.incj, %for.i
  %j = phi i64 [ %incj, %for.incj ], [ 0, %for.i ]
  %mul7 = mul nsw i64 %j, 3
  %tmp = add i64 %sub, %mul7
  %tmp27 = mul i64 %tmp, %o
  br label %for.k

for.k:                                            ; preds = %for.k, %for.j.us
  %k = phi i64 [ 0, %for.j ], [ %inck, %for.k ]

  %mul = mul nsw i64 %k, 2
  %arrayidx.sum = add i64 %mul, 7
  %arrayidx10.sum = add i64 %arrayidx.sum, %tmp27
  %arrayidx11 = getelementptr inbounds i32, i32* %A, i64 %arrayidx10.sum
  store i32 1, i32* %arrayidx11, align 4

  %inck = add nsw i64 %k, 1
  %exitcond.us = icmp eq i64 %inck, %o
  br i1 %exitcond.us, label %for.incj, label %for.k

for.incj:                                         ; preds = %for.k
  %incj = add nsw i64 %j, 1
  %exitcond54.us = icmp eq i64 %incj, %m
  br i1 %exitcond54.us, label %for.inci, label %for.j

for.inci:                                         ; preds = %for.incj
  %inci = add nsw i64 %i, 1
  %exitcond55.us = icmp eq i64 %inci, %n
  br i1 %exitcond55.us, label %for.end.loopexit, label %for.i

for.end.loopexit:                                 ; preds = %for.inci
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %for.cond1.preheader.lr.ph, %entry
  ret void
}

; Loop i is supposed to have the largest cost and be placed
; as the outermost loop. This test differs from foo() since the
; loopnest has a suboptimal order j-i-k.
; After D123400 we ensure that the order of loop cache analysis output
; is loop i-j-k, despite the suboptimal order in the original loopnest.
;
; void foo(long n, long m, long o, int A[n][m][o]) {
;   for (long j = 0; j < m; j++)
;     for (long i = 0; i < n; i++)
;       for (long k = 0; k < o; k++)
;         A[2*i+3][2*j-4][2*k+7] = 1;
; }

; CHECK: Loop 'for.i' has cost = 100000000
; CHECK-NEXT: Loop 'for.j' has cost = 1000000
; CHECK-NEXT: Loop 'for.k' has cost = 60000

define void @foo2(i64 %n, i64 %m, i64 %o, i32* %A) {
entry:
  %cmp32 = icmp sgt i64 %n, 0
  %cmp230 = icmp sgt i64 %m, 0
  %cmp528 = icmp sgt i64 %o, 0
  br i1 %cmp32, label %for.cond1.preheader.lr.ph, label %for.end

for.cond1.preheader.lr.ph:                        ; preds = %entry
  br i1 %cmp230, label %for.j.preheader, label %for.end

for.j.preheader:                                  ; preds = %for.cond1.preheader.lr.ph
  br i1 %cmp528, label %for.j.preheader.split, label %for.end

for.j.preheader.split:                            ; preds = %for.j.preheader
  br label %for.j

for.i:                                            ; preds = %for.inci, %for.j
  %i = phi i64 [ %inci, %for.inci ], [ 0, %for.j ]
  %mul8 = shl i64 %i, 1
  %add9 = add nsw i64 %mul8, 3
  %0 = mul i64 %add9, %m
  %sub = add i64 %0, -4
  %mul7 = mul nsw i64 %j, 2
  %tmp = add i64 %sub, %mul7
  %tmp27 = mul i64 %tmp, %o
  br label %for.k

for.j:                                            ; preds = %for.incj, %for.j.preheader.split
  %j = phi i64 [ %incj, %for.incj ], [ 0, %for.j.preheader.split ]
  br label %for.i

for.k:                                            ; preds = %for.k, %for.i
  %k = phi i64 [ 0, %for.i ], [ %inck, %for.k ]

  %mul = mul nsw i64 %k, 2
  %arrayidx.sum = add i64 %mul, 7
  %arrayidx10.sum = add i64 %arrayidx.sum, %tmp27
  %arrayidx11 = getelementptr inbounds i32, i32* %A, i64 %arrayidx10.sum
  store i32 1, i32* %arrayidx11, align 4

  %inck = add nsw i64 %k, 1
  %exitcond.us = icmp eq i64 %inck, %o
  br i1 %exitcond.us, label %for.inci, label %for.k

for.incj:                                         ; preds = %for.inci
  %incj = add nsw i64 %j, 1
  %exitcond54.us = icmp eq i64 %incj, %m
  br i1 %exitcond54.us, label %for.end.loopexit, label %for.j

for.inci:                                         ; preds = %for.k
  %inci = add nsw i64 %i, 1
  %exitcond55.us = icmp eq i64 %inci, %n
  br i1 %exitcond55.us, label %for.incj, label %for.i

for.end.loopexit:                                 ; preds = %for.incj
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %for.cond1.preheader.lr.ph, %entry
  ret void
}
