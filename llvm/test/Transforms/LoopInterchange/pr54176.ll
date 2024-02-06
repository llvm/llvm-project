; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 -verify-dom-info -verify-loop-info \
; RUN:     -S -debug 2>&1 | FileCheck %s

@bb = global [1024 x [128 x float]] zeroinitializer, align 4
@aa = global [1024 x [128 x float]] zeroinitializer, align 4
@cc = global [1024 x [128 x float]] zeroinitializer, align 4


;;  for (int j = 1; j < M; j++)
;;    for (int i = 1; i < N; i++) {
;;      aa[1][j-1] += bb[i][j];
;;      cc[i][j] = aa[1][j];
;;    }

; CHECK: Has constant index with loop carried dependencies inside loop
; CHECK: Populating dependency matrix failed

define void @pr54176() {
entry:
  br label %for.cond1.preheader

; Loop:
for.cond1.preheader:                              ; preds = %entry, %for.cond.cleanup3
  %indvars.iv28 = phi i64 [ 1, %entry ], [ %indvars.iv.next29, %for.cond.cleanup3 ]
  %0 = add nsw i64 %indvars.iv28, -1
  %arrayidx8 = getelementptr inbounds [1024 x [128 x float]], ptr @aa, i64 0, i64 1, i64 %0
  %arrayidx10 = getelementptr inbounds [1024 x [128 x float]], ptr @aa, i64 0, i64 1, i64 %indvars.iv28
  br label %for.body4

for.cond.cleanup3:                                ; preds = %for.body4
  %indvars.iv.next29 = add nuw nsw i64 %indvars.iv28, 1
  %exitcond31 = icmp ne i64 %indvars.iv.next29, 128
  br i1 %exitcond31, label %for.cond1.preheader, label %for.cond.cleanup

for.body4:                                        ; preds = %for.cond1.preheader, %for.body4
  %indvars.iv = phi i64 [ 1, %for.cond1.preheader ], [ %indvars.iv.next, %for.body4 ]
  %arrayidx6 = getelementptr inbounds [1024 x [128 x float]], ptr @bb, i64 0, i64 %indvars.iv, i64 %indvars.iv28
  %1 = load float, ptr %arrayidx6, align 4
  %2 = load float, ptr %arrayidx8, align 4
  %add = fadd float %1, %2
  store float %add, ptr %arrayidx8, align 4
  %3 = load float, ptr %arrayidx10, align 4
  %arrayidx14 = getelementptr inbounds [1024 x [128 x float]], ptr @cc, i64 0, i64 %indvars.iv, i64 %indvars.iv28
  store float %3, ptr %arrayidx14, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.body4, label %for.cond.cleanup3

; Exit blocks
for.cond.cleanup:                                 ; preds = %for.cond.cleanup3
  ret void
}
