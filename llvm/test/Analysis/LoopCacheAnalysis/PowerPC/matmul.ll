; RUN: opt < %s -passes='print<loop-cache-cost>' -disable-output 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; void matmul(long n, long m, long o, int A[n][m], int B[n][m], int C[n]) {
;   for (long i = 0; i < n; i++) 
;     for (long j = 0; j < m; j++) 
;       for (long k = 0; k < o; k++) 
;         C[i][j] = C[i][j] + A[i][k] * B[k][j];
; }

; CHECK:Loop 'for.i' has cost = 2010000
; CHECK:Loop 'for.k' has cost = 1040000
; CHECK:Loop 'for.j' has cost = 70000
    
define void @matmul(i64 %n, i64 %m, i64 %o, i32* %A, i32* %B, i32* %C) {
entry:
  br label %for.i

for.i:                                         ; preds = %entry, %for.inc.i
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.inc.i ]
  %muli = mul i64 %i, %m
  br label %for.j

for.j:                                        ; preds = %for.i, %for.inc.j
  %j = phi i64 [ 0, %for.i ], [ %j.next, %for.inc.j ]
  %addj = add i64 %muli, %j
  %mulj = mul i64 %addj, %o
  br label %for.k

for.k:                                        ; preds = %for.j, %for.inc.k
  %k = phi i64 [ 0, %for.j ], [ %k.next, %for.inc.k ]

  ; A[i][k]
  %arrayidx3 = add i64 %k, %muli
  %arrayidx4 = getelementptr inbounds i32, i32* %A, i64 %arrayidx3
  %elem_A = load i32, i32* %arrayidx4, align 4

  ; B[k][j]
  %mulk = mul i64 %k, %o
  %arrayidx5 = add i64 %j, %mulk
  %arrayidx6 = getelementptr inbounds i32, i32* %B, i64 %arrayidx5
  %elem_B = load i32, i32* %arrayidx6, align 4

  ; C[i][k]
  %arrayidx7 = add i64 %j, %muli
  %arrayidx8 = getelementptr inbounds i32, i32* %C, i64 %arrayidx7
  %elem_C = load i32, i32* %arrayidx8, align 4

  ; C[i][j] = C[i][j] + A[i][k] * B[k][j];
  %mul = mul nsw i32 %elem_A, %elem_B
  %add = add nsw i32 %elem_C, %mul
  store i32 %add, i32* %arrayidx8, align 4

  br label %for.inc.k

for.inc.k:                                          ; preds = %for.k
  %k.next = add nuw nsw i64 %k, 1
  %exitcond = icmp ne i64 %k.next, %o
  br i1 %exitcond, label %for.k, label %for.end

for.end:                                          ; preds = %for.inc
  br label %for.inc.j

for.inc.j:                                        ; preds = %for.end
  %j.next = add nuw nsw i64 %j, 1
  %exitcond5 = icmp ne i64 %j.next, %m
  br i1 %exitcond5, label %for.j, label %for.end23

for.end23:                                        ; preds = %for.inc.j
  br label %for.inc.i

for.inc.i:                                        ; preds = %for.end23
  %i.next = add nuw nsw i64 %i, 1
  %exitcond8 = icmp ne i64 %i.next, %n
  br i1 %exitcond8, label %for.i, label %for.end26

for.end26:                                        ; preds = %for.inc.i
  ret void
}
