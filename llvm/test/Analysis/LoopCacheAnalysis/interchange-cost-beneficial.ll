; RUN: opt <  %s  -cache-line-size=64 -passes='print<loop-cache-cost>' -disable-output 2>&1 | FileCheck  %s

;; This test checks the effect of rounding cache cost to 1 when it is 
;; evaluated to 0 because at least 1 cache line is accessed by the loopnest.
;; It does not make sense to output that zero cache lines are used.
;; The cost of reference group for B[j], C[j], D[j] and E[j] were 
;; calculted 0 before but now they are 1 which makes each loop cost more reasonable.
;
; void test(int n, int m, int o, int A[2][3], int B[2], int C[2], int D[2], int E[2]) {
;   for (int i = 0; i < 3; i++)
;     for (int j = 0; j < 2; j++)
;        A[j][i] = 1;
;        B[j] = 1;
;        C[j] = 1;
;        D[j] = 1
;        E[j] = 1
; }

; CHECK: Loop 'for.j' has cost = 18
; CHECK-NEXT: Loop 'for.i' has cost = 10

define void @test(ptr %A, ptr %B, ptr %C, ptr %D, ptr %E) {

entry:
  br label %for.i.preheader.split

for.i.preheader.split:                            ; preds = %for.i.preheader
  br label %for.i

for.i:                                            ; preds = %for.inci, %for.i.preheader.split
  %i = phi i64 [ %inci, %for.inci ], [ 0, %for.i.preheader.split ]
  br label %for.j

for.j:                                            ; preds = %for.incj, %for.i
  %j = phi i64 [ %incj, %for.j ], [ 0, %for.i ]
  %mul_j = mul nsw i64 %j, 3
  %index_j = add i64 %mul_j, %i
  %arrayidxA = getelementptr inbounds [2 x [ 3 x i32]], ptr %A, i64 %j, i64 %i
  store i32 1, ptr %arrayidxA, align 4
  %arrayidxB = getelementptr inbounds i32, ptr %B, i64 %j
  store i32 1, ptr %arrayidxB, align 4
  %arrayidxC = getelementptr inbounds i32, ptr %C, i64 %j
  store i32 1, ptr %arrayidxC, align 4
  %arrayidxD = getelementptr inbounds i32, ptr %D, i64 %j
  store i32 1, ptr %arrayidxD, align 4
  %arrayidxE = getelementptr inbounds i32, ptr %E, i64 %j
  store i32 1, ptr %arrayidxE, align 4
  %incj = add nsw i64 %j, 1
  %exitcond.us = icmp eq i64 %incj, 2
  br i1 %exitcond.us, label %for.inci, label %for.j

for.inci:                                         ; preds = %for.incj
  %inci = add nsw i64 %i, 1
  %exitcond55.us = icmp eq i64 %inci, 3
  br i1 %exitcond55.us, label %for.end.loopexit, label %for.i

for.end.loopexit:                                 ; preds = %for.inci
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %for.cond1.preheader.lr.ph, %entry
  ret void
}
