; RUN: opt %loadNPMPolly '-passes=polly-custom<simplify-0;optree;delicm;simplify-1;opt-isl;ast>' -plugin-arg=Polly,-polly-print-ast -disable-output < %s | FileCheck %s
; RUN: opt %loadNPMPolly '-passes=polly-custom<simplify-0;optree;delicm;simplify-1;opt-isl;ast>' -plugin-arg=Polly,-polly-print-ast -plugin-arg=Polly,-polly-tile-outer-bands=false -disable-output < %s | FileCheck %s --check-prefix=INNERMOST
; RUN: opt %loadNPMPolly '-passes=polly-custom<simplify-0;optree;delicm;simplify-1;opt-isl;ast>' -plugin-arg=Polly,-polly-print-ast -plugin-arg=Polly,-polly-register-tiling -disable-output < %s | FileCheck %s
; RUN: opt %loadNPMPolly '-passes=polly-custom<simplify-0;optree;delicm;simplify-1;opt-isl;ast>' -plugin-arg=Polly,-polly-print-ast -plugin-arg=Polly,-polly-vectorizer=stripmine -disable-output < %s | FileCheck %s --check-prefix=VECT
;
; Tile a permutable band even though it contains another loop.
;
;    void trmm(int m, int n, float alpha, float A[restrict m][m],
;              float B[restrict m][n]) {
;      for (int i = 0; i < m; i++)
;        for (int j = 0; j < n; j++) {
;          for (int k = i + 1; k < m; k++)
;            B[i][j] += A[k][i] * B[k][j];
;          B[i][j] = alpha * B[i][j];
;        }
;    }
;
; After DeLICM has mapped the accumulator to B[i][j], the scheduler puts the
; loops over j and i + k into a permutable band whose child is the band of the
; triangular loop over i. Only innermost bands used to be tiled, which left the
; main loop nest untiled.
;
; Register tiling and prevectorization must leave the outer point loops intact.
; VECT: // 1st level tiling - Tiles
; VECT: for (int c0 = 0; c0 <= floord(n - 1, 32); c0 += 1)
; VECT: for (int c1 = 0; c1 <= floord(m - 2, 32); c1 += 1) {
; VECT: // 1st level tiling - Points
; VECT-NEXT: for (int c2 = 0; c2 <= min(31, n - 32 * c0 - 1); c2 += 1)

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

define void @trmm(i32 %m, i32 %n, float %alpha, ptr noalias %A, ptr noalias %B) {
entry:
  %0 = zext i32 %m to i64
  %1 = zext i32 %n to i64
  %cmp62 = icmp sgt i32 %m, 0
  %cmp260 = icmp sgt i32 %n, 0
  %or.cond = and i1 %cmp62, %cmp260
  br i1 %or.cond, label %for.cond1.preheader.preheader, label %for.cond.cleanup

for.cond1.preheader.preheader:
  %2 = zext nneg i32 %m to i64
  br label %for.cond1.preheader

for.cond1.preheader:
  %indvars.iv78 = phi i64 [ 0, %for.cond1.preheader.preheader ], [ %indvars.iv.next79, %for.cond1.for.cond.cleanup3_crit_edge ]
  %indvars.iv66 = phi i64 [ 1, %for.cond1.preheader.preheader ], [ %indvars.iv.next67, %for.cond1.for.cond.cleanup3_crit_edge ]
  %indvars.iv.next79 = add nuw nsw i64 %indvars.iv78, 1
  %cmp656 = icmp samesign ult i64 %indvars.iv.next79, %2
  %invariant.gep = getelementptr inbounds nuw [4 x i8], ptr %A, i64 %indvars.iv78
  %3 = mul nuw nsw i64 %indvars.iv78, %1
  %arrayidx16 = getelementptr inbounds nuw [4 x i8], ptr %B, i64 %3
  br i1 %cmp656, label %for.cond5.preheader.us, label %for.cond5.preheader

for.cond5.preheader.us:
  %indvars.iv73 = phi i64 [ %indvars.iv.next74, %for.cond5.for.cond.cleanup7_crit_edge.us ], [ 0, %for.cond1.preheader ]
  %invariant.gep58.us = getelementptr inbounds nuw [4 x i8], ptr %B, i64 %indvars.iv73
  %arrayidx18.us = getelementptr inbounds nuw [4 x i8], ptr %arrayidx16, i64 %indvars.iv73
  %arrayidx18.promoted.us = load float, ptr %arrayidx18.us, align 4
  br label %for.body8.us

for.body8.us:
  %indvars.iv68 = phi i64 [ %indvars.iv66, %for.cond5.preheader.us ], [ %indvars.iv.next69, %for.body8.us ]
  %4 = phi float [ %arrayidx18.promoted.us, %for.cond5.preheader.us ], [ %9, %for.body8.us ]
  %5 = mul nuw nsw i64 %indvars.iv68, %0
  %gep.us = getelementptr inbounds nuw [4 x i8], ptr %invariant.gep, i64 %5
  %6 = load float, ptr %gep.us, align 4
  %7 = mul nuw nsw i64 %indvars.iv68, %1
  %gep59.us = getelementptr inbounds nuw [4 x i8], ptr %invariant.gep58.us, i64 %7
  %8 = load float, ptr %gep59.us, align 4
  %9 = tail call float @llvm.fmuladd.f32(float %6, float %8, float %4)
  store float %9, ptr %arrayidx18.us, align 4
  %indvars.iv.next69 = add nuw nsw i64 %indvars.iv68, 1
  %exitcond72.not = icmp eq i64 %indvars.iv.next69, %0
  br i1 %exitcond72.not, label %for.cond5.for.cond.cleanup7_crit_edge.us, label %for.body8.us

for.cond5.for.cond.cleanup7_crit_edge.us:
  %mul.us = fmul float %alpha, %9
  store float %mul.us, ptr %arrayidx18.us, align 4
  %indvars.iv.next74 = add nuw nsw i64 %indvars.iv73, 1
  %exitcond77.not = icmp eq i64 %indvars.iv.next74, %1
  br i1 %exitcond77.not, label %for.cond1.for.cond.cleanup3_crit_edge, label %for.cond5.preheader.us

for.cond.cleanup:
  ret void

for.cond5.preheader:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.cond5.preheader ], [ 0, %for.cond1.preheader ]
  %arrayidx22 = getelementptr inbounds nuw [4 x i8], ptr %arrayidx16, i64 %indvars.iv
  %10 = load float, ptr %arrayidx22, align 4
  %mul = fmul float %alpha, %10
  store float %mul, ptr %arrayidx22, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %1
  br i1 %exitcond.not, label %for.cond1.for.cond.cleanup3_crit_edge, label %for.cond5.preheader

for.cond1.for.cond.cleanup3_crit_edge:
  %indvars.iv.next67 = add nuw nsw i64 %indvars.iv66, 1
  %exitcond82.not = icmp eq i64 %indvars.iv.next79, %0
  br i1 %exitcond82.not, label %for.cond.cleanup, label %for.cond1.preheader
}


; CHECK:      // 1st level tiling - Tiles
; CHECK-NEXT: for (int c0 = 0; c0 <= floord(n - 1, 32); c0 += 1)
; CHECK-NEXT:   for (int c1 = 0; c1 <= floord(m - 2, 32); c1 += 1) {
; CHECK-NEXT:     // 1st level tiling - Points
; CHECK-NEXT:     for (int c2 = 0; c2 <= min(31, n - 32 * c0 - 1); c2 += 1)
; CHECK-NEXT:       for (int c3 = 0; c3 <= min(31, m - 32 * c1 - 2); c3 += 1)
; CHECK-NEXT:         for (int c4 = 0; c4 <= 32 * c1 + c3; c4 += 1)
; CHECK-NEXT:           Stmt_for_body8_us(c4, 32 * c0 + c2, 32 * c1 + c3 - c4);
; CHECK-NEXT:   }

; INNERMOST:      for (int c0 = 0; c0 < n; c0 += 1)
; INNERMOST-NEXT:   for (int c1 = 0; c1 < m - 1; c1 += 1)
; INNERMOST-NEXT:     for (int c2 = 0; c2 <= c1; c2 += 1)
; INNERMOST-NEXT:       Stmt_for_body8_us(c2, c0, c1 - c2);
