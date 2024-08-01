; RUN: opt %loadNPMPolly -polly-region-expansion-profitability-check=1 \
; RUN: -polly-loopfusion-greedy=1 "-passes=scop(polly-opt-isl,print<polly-ast>)" \
; RUN: -disable-output < %s | FileCheck %s

; void foo(int *restrict A, int *restrict B, int *restrict C, int n, int *restrict U) {
;   for (int i = 0; i < 1024; ++i) // L1
;     B[i] += A[i];
;   int u = *U; // LICM
;   for (int j = 0; j < 1024; ++j) // L2
;     C[j] = B[j] + u;
; }

define void @foo(ptr noalias nocapture noundef readonly %A, ptr noalias nocapture noundef %B, ptr noalias nocapture noundef writeonly %C, i32 noundef %n, ptr noalias nocapture noundef readonly %U) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %entry.split, %for.body
  %indvars.iv = phi i64 [ 0, %entry.split ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %i1 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %B, i64 %indvars.iv
  %i2 = load i32, ptr %arrayidx2, align 4
  %add = add nsw i32 %i2, %i1
  store i32 %add, ptr %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond3.preheader, label %for.body

for.cond3.preheader:                              ; preds = %for.body
  %i3 = load i32, ptr %U, align 4
  br label %for.body6

for.body6:                                        ; preds = %for.cond3.preheader, %for.body6
  %indvars.iv25 = phi i64 [ 0, %for.cond3.preheader ], [ %indvars.iv.next26, %for.body6 ]
  %arrayidx8 = getelementptr inbounds i32, ptr %B, i64 %indvars.iv25
  %i4 = load i32, ptr %arrayidx8, align 4
  %add9 = add nsw i32 %i3, %i4
  %arrayidx11 = getelementptr inbounds i32, ptr %C, i64 %indvars.iv25
  store i32 %add9, ptr %arrayidx11, align 4
  %indvars.iv.next26 = add nuw nsw i64 %indvars.iv25, 1
  %exitcond28.not = icmp eq i64 %indvars.iv.next26, 1024
  br i1 %exitcond28.not, label %for.cond.cleanup5, label %for.body6

for.cond.cleanup5:                                ; preds = %for.body6
  ret void
}

; Check that loop preheader block that containing loop invariant load *U is
; handled properly by region expansion profitability check, L1 and L2 are fused.
; CHECK-LABEL: isl ast :: foo :: %for.body---%for.cond.cleanup5
; CHECK:           Stmt_for_cond3_preheader();
; CHECK-NEXT:      for (int c0 = 0; c0 <= 1023; c0 += 1) {
; CHECK-NEXT:        Stmt_for_body(c0);
; CHECK-NEXT:        Stmt_for_body6(c0);
