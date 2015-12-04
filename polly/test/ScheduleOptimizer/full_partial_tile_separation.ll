; RUN: opt -S %loadPolly -polly-vectorizer=stripmine -polly-opt-isl -polly-ast -analyze < %s | FileCheck %s
; CHECK:       // 1st level tiling - Tiles
; CHECK:       #pragma known-parallel
; CHECK:       for (int c0 = 0; c0 <= floord(ni - 1, 32); c0 += 1)
; CHECK:         for (int c1 = 0; c1 <= floord(nj - 1, 32); c1 += 1)
; CHECK:           for (int c2 = 0; c2 <= floord(nk - 1, 32); c2 += 1) {
; CHECK:             // 1st level tiling - Points
; CHECK:             for (int c3 = 0; c3 <= min(31, ni - 32 * c0 - 1); c3 += 1) {
; CHECK:               for (int c4 = 0; c4 <= min(7, -8 * c1 + nj / 4 - 1); c4 += 1)
; CHECK:                 for (int c5 = 0; c5 <= min(31, nk - 32 * c2 - 1); c5 += 1)
; CHECK:                   #pragma simd
; CHECK:                   for (int c6 = 0; c6 <= 3; c6 += 1)
; CHECK:                     Stmt_for_body_6(32 * c0 + c3, 32 * c1 + 4 * c4 + c6, 32 * c2 + c5);
; CHECK:               if (32 * c1 + 31 >= nj)
; CHECK:                 for (int c5 = 0; c5 <= min(31, nk - 32 * c2 - 1); c5 += 1)
; CHECK:                   #pragma simd
; CHECK:                   for (int c6 = 0; c6 < nj % 4; c6 += 1)
; CHECK:                     Stmt_for_body_6(32 * c0 + c3, -((nj - 1) % 4) + nj + c6 - 1, 32 * c2 + c5);
; CHECK:             }
; CHECK:           }

; Function Attrs: nounwind uwtable
define void @kernel_gemm(i32 %ni, i32 %nj, i32 %nk, double %alpha, double %beta, [1024 x double]* %C, [1024 x double]* %A, [1024 x double]* %B) #0 {
entry:
  %cmp.27 = icmp sgt i32 %ni, 0
  br i1 %cmp.27, label %for.cond.1.preheader.lr.ph, label %for.end.22

for.cond.1.preheader.lr.ph:                       ; preds = %entry
  br label %for.cond.1.preheader

for.cond.1.preheader:                             ; preds = %for.cond.1.preheader.lr.ph, %for.inc.20
  %indvars.iv33 = phi i64 [ 0, %for.cond.1.preheader.lr.ph ], [ %indvars.iv.next34, %for.inc.20 ]
  %cmp2.25 = icmp sgt i32 %nj, 0
  br i1 %cmp2.25, label %for.cond.4.preheader.lr.ph, label %for.inc.20

for.cond.4.preheader.lr.ph:                       ; preds = %for.cond.1.preheader
  br label %for.cond.4.preheader

for.cond.4.preheader:                             ; preds = %for.cond.4.preheader.lr.ph, %for.inc.17
  %indvars.iv29 = phi i64 [ 0, %for.cond.4.preheader.lr.ph ], [ %indvars.iv.next30, %for.inc.17 ]
  %cmp5.23 = icmp sgt i32 %nk, 0
  br i1 %cmp5.23, label %for.body.6.lr.ph, label %for.inc.17

for.body.6.lr.ph:                                 ; preds = %for.cond.4.preheader
  br label %for.body.6

for.body.6:                                       ; preds = %for.body.6.lr.ph, %for.body.6
  %indvars.iv = phi i64 [ 0, %for.body.6.lr.ph ], [ %indvars.iv.next, %for.body.6 ]
  %arrayidx8 = getelementptr inbounds [1024 x double], [1024 x double]* %A, i64 %indvars.iv33, i64 %indvars.iv
  %0 = load double, double* %arrayidx8, align 8
  %arrayidx12 = getelementptr inbounds [1024 x double], [1024 x double]* %B, i64 %indvars.iv, i64 %indvars.iv29
  %1 = load double, double* %arrayidx12, align 8
  %mul = fmul double %0, %1
  %arrayidx16 = getelementptr inbounds [1024 x double], [1024 x double]* %C, i64 %indvars.iv33, i64 %indvars.iv29
  %2 = load double, double* %arrayidx16, align 8
  %add = fadd double %2, %mul
  store double %add, double* %arrayidx16, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, %nk
  br i1 %exitcond, label %for.body.6, label %for.cond.4.for.inc.17_crit_edge

for.cond.4.for.inc.17_crit_edge:                  ; preds = %for.body.6
  br label %for.inc.17

for.inc.17:                                       ; preds = %for.cond.4.for.inc.17_crit_edge, %for.cond.4.preheader
  %indvars.iv.next30 = add nuw nsw i64 %indvars.iv29, 1
  %lftr.wideiv31 = trunc i64 %indvars.iv.next30 to i32
  %exitcond32 = icmp ne i32 %lftr.wideiv31, %nj
  br i1 %exitcond32, label %for.cond.4.preheader, label %for.cond.1.for.inc.20_crit_edge

for.cond.1.for.inc.20_crit_edge:                  ; preds = %for.inc.17
  br label %for.inc.20

for.inc.20:                                       ; preds = %for.cond.1.for.inc.20_crit_edge, %for.cond.1.preheader
  %indvars.iv.next34 = add nuw nsw i64 %indvars.iv33, 1
  %lftr.wideiv35 = trunc i64 %indvars.iv.next34 to i32
  %exitcond36 = icmp ne i32 %lftr.wideiv35, %ni
  br i1 %exitcond36, label %for.cond.1.preheader, label %for.cond.for.end.22_crit_edge

for.cond.for.end.22_crit_edge:                    ; preds = %for.inc.20
  br label %for.end.22

for.end.22:                                       ; preds = %for.cond.for.end.22_crit_edge, %entry
  ret void
}
