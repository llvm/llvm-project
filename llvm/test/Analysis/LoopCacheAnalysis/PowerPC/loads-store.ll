; RUN: opt < %s -passes='print<loop-cache-cost>' -disable-output 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; void foo(long n, long m, long o, int A[n][m][o], int B[n][m][o], int C[n][m][o]) {
;   for (long i = 0; i < n; i++)
;     for (long j = 0; j < m; j++)
;       for (long k = 0; k < o; k++)
;         A[i][k][j] += B[i][k][j] + C[i][j][k];
; }

; CHECK: Loop 'for.i' has cost = 3000000
; CHECK-NEXT: Loop 'for.k' has cost = 2030000
; CHECK-NEXT: Loop 'for.j' has cost = 1060000

define void @foo(i64 %n, i64 %m, i64 %o, ptr %A, ptr %B, ptr %C) {
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
  %muli = mul i64 %i, %m
  br label %for.j

for.j:                                            ; preds = %for.incj, %for.i
  %j = phi i64 [ %incj, %for.incj ], [ 0, %for.i ]
  %addj = add i64 %muli, %j
  %mulj = mul i64 %addj, %o
  br label %for.k

for.k:                                            ; preds = %for.k, %for.j
  %k = phi i64 [ 0, %for.j ], [ %inck, %for.k ]

  ; B[i][k][j]
  %addk = add i64 %muli, %k
  %mulk = mul i64 %addk, %o
  %arrayidx1 = add i64 %j, %mulk
  %arrayidx2 = getelementptr inbounds i32, ptr %B, i64 %arrayidx1
  %elem_B = load i32, ptr %arrayidx2, align 4

  ; C[i][j][k]
  %arrayidx3 = add i64 %k, %mulj
  %arrayidx4 = getelementptr inbounds i32, ptr %C, i64 %arrayidx3
  %elem_C = load i32, ptr %arrayidx4, align 4

  ; A[i][k][j]
  %arrayidx5 = getelementptr inbounds i32, ptr %A, i64 %arrayidx1
  %elem_A = load i32, ptr %arrayidx5, align 4

  ; A[i][k][j] += B[i][k][j] + C[i][j][k]
  %add1 = add i32 %elem_B, %elem_C
  %add2 = add i32 %add1, %elem_A
  %arrayidx6 = getelementptr inbounds i32, ptr %A, i64 %arrayidx1
  store i32 %add2, ptr %arrayidx6, align 4

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
