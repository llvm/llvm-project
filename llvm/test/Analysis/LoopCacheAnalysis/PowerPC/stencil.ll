; RUN: opt < %s -passes='print<loop-cache-cost>' -disable-output 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; void foo(long n, long m, long o, int A[n][m], int B[n][m], int C[n]) {
;   for (long i = 0; i < n; i++)
;     for (long j = 0; j < m; j++) {
;         A[i][j] = A[i][j+1] + B[i-1][j] + B[i+1][j+1] + C[i];
;         A[i][j] += B[i][i];
;     }   
; }

; CHECK: Loop 'for.i' has cost = 20600
; CHECK-NEXT: Loop 'for.j' has cost = 800

define void @foo(i64 %n, i64 %m, i32* %A, i32* %B, i32* %C) {
entry:
  %cmp32 = icmp sgt i64 %n, 0
  %cmp230 = icmp sgt i64 %m, 0
  br i1 %cmp32, label %for.cond1.preheader.lr.ph, label %for.end

for.cond1.preheader.lr.ph:                        ; preds = %entry
  br i1 %cmp230, label %for.i.preheader, label %for.end

for.i.preheader:                                  ; preds = %for.cond1.preheader.lr.ph
  br label %for.i

for.i:                                            ; preds = %for.inci, %for.i.preheader.split
  %i = phi i64 [ %inci, %for.inci ], [ 0, %for.i.preheader ]
  %subione = sub i64 %i, 1
  %addione = add i64 %i, 1
  %muli = mul i64 %i, %m
  %muliminusone = mul i64 %subione, %m
  %muliplusone = mul i64 %addione, %m
  br label %for.j

for.j:                                            ; preds = %for.incj, %for.i
  %j = phi i64 [ %incj, %for.incj ], [ 0, %for.i ]
  %addj = add i64 %muli, %j

  ; B[i-1][j]
  %arrayidx1 = add i64 %j, %muliminusone
  %arrayidx2 = getelementptr inbounds i32, i32* %B, i64 %arrayidx1
  %elem_B1 = load i32, i32* %arrayidx2, align 4

  ; B[i-1][j+1]
  %addjone = add i64 %j, 1
  %arrayidx3 = add i64 %addjone, %muliminusone
  %arrayidx4 = getelementptr inbounds i32, i32* %B, i64 %arrayidx3
  %elem_B2 = load i32, i32* %arrayidx4, align 4

  ; C[i]
  %arrayidx6 = getelementptr inbounds i32, i32* %C, i64 %i
  %elem_C = load i32, i32* %arrayidx6, align 4

  ; A[i][j+1]
  %arrayidx7 = add i64 %addjone, %muli
  %arrayidx8 = getelementptr inbounds i32, i32* %A, i64 %arrayidx7
  %elem_A = load i32, i32* %arrayidx8, align 4

  ; A[i][j] = A[i][j+1] + B[i-1][j] + B[i-1][j+1] + C[i]
  %addB = add i32 %elem_B1, %elem_B2
  %addC = add i32 %addB, %elem_C
  %addA = add i32 %elem_A, %elem_C
  %arrayidx9 = add i64 %j, %muli
  %arrayidx10 = getelementptr inbounds i32, i32* %A, i64 %arrayidx9
  store i32 %addA, i32* %arrayidx10, align 4

  ; A[i][j] += B[i][i];
  %arrayidx11 = add i64 %j, %muli
  %arrayidx12 = getelementptr inbounds i32, i32* %A, i64 %arrayidx11
  %elem_A1 = load i32, i32* %arrayidx12, align 4
  %arrayidx13 = add i64 %i, %muli
  %arrayidx14 = getelementptr inbounds i32, i32* %B, i64 %arrayidx13
  %elem_B3 = load i32, i32* %arrayidx14, align 4
  %addA1 = add i32 %elem_A1, %elem_B3
  store i32 %addA1, i32* %arrayidx12, align 4

  br label %for.incj

for.incj:                                         ; preds = %for.j
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

