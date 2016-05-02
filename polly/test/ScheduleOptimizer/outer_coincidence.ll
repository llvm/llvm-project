; RUN: opt %loadPolly -polly-opt-isl -polly-ast -polly-tiling=0 -polly-parallel -polly-opt-outer-coincidence=no -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -polly-opt-isl -polly-ast -polly-tiling=0 -polly-parallel -polly-opt-outer-coincidence=yes -analyze < %s | FileCheck %s --check-prefix=OUTER

; By skewing, the diagonal can be made parallel. ISL does this when the Check
; the 'outer_coincidence' option is enabled.
;
; void func(int m, int n, float A[static const restrict m][n]) {
;  for (int i = 1; i < m; i+=1)
;    for (int j = 1; j < n; j+=1)
;      A[i][j] = A[i-1][j] + A[i][j-1];
;}

define void @func(i64 %m, i64 %n, float* noalias nonnull %A) #0 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc11, %entry
  %i.0 = phi i64 [ 1, %entry ], [ %add12, %for.inc11 ]
  %cmp = icmp slt i64 %i.0, %m
  br i1 %cmp, label %for.cond1.preheader, label %for.end13

for.cond1.preheader:                              ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.cond1.preheader, %for.body3
  %j.0 = phi i64 [ %add10, %for.body3 ], [ 1, %for.cond1.preheader ]
  %cmp2 = icmp slt i64 %j.0, %n
  br i1 %cmp2, label %for.body3, label %for.inc11

for.body3:                                        ; preds = %for.cond1
  %sub = add nsw i64 %i.0, -1
  %tmp = mul nsw i64 %sub, %n
  %arrayidx = getelementptr inbounds float, float* %A, i64 %tmp
  %arrayidx4 = getelementptr inbounds float, float* %arrayidx, i64 %j.0
  %tmp13 = load float, float* %arrayidx4, align 4
  %sub5 = add nsw i64 %j.0, -1
  %tmp14 = mul nsw i64 %i.0, %n
  %arrayidx6 = getelementptr inbounds float, float* %A, i64 %tmp14
  %arrayidx7 = getelementptr inbounds float, float* %arrayidx6, i64 %sub5
  %tmp15 = load float, float* %arrayidx7, align 4
  %add = fadd float %tmp13, %tmp15
  %tmp16 = mul nsw i64 %i.0, %n
  %arrayidx8 = getelementptr inbounds float, float* %A, i64 %tmp16
  %arrayidx9 = getelementptr inbounds float, float* %arrayidx8, i64 %j.0
  store float %add, float* %arrayidx9, align 4
  %add10 = add nuw nsw i64 %j.0, 1
  br label %for.cond1

for.inc11:                                        ; preds = %for.cond1
  %add12 = add nuw nsw i64 %i.0, 1
  br label %for.cond

for.end13:                                        ; preds = %for.cond
  ret void
}


; CHECK:      #pragma minimal dependence distance: 1
; CHECK-NEXT: for (int c0 = 0; c0 < m - 1; c0 += 1)
; CHECK-NEXT:   #pragma minimal dependence distance: 1
; CHECK-NEXT:   for (int c1 = 0; c1 < n - 1; c1 += 1)
; CHECK-NEXT:     Stmt_for_body3(c0, c1);

; OUTER:      #pragma minimal dependence distance: 1
; OUTER-NEXT: for (int c0 = 0; c0 < m + n - 3; c0 += 1)
; OUTER-NEXT:   #pragma simd
; OUTER-NEXT:   #pragma known-parallel
; OUTER-NEXT:   for (int c1 = max(0, -m + c0 + 2); c1 <= min(n - 2, c0); c1 += 1)
; OUTER-NEXT:     Stmt_for_body3(c0 - c1, c1);
