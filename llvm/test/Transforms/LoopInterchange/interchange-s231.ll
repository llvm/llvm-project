; REQUIRES: asserts
; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 -verify-dom-info -verify-loop-info \
; RUN:     -S -debug 2>&1 | FileCheck %s

@aa = global [256 x [256 x float]] zeroinitializer, align 64
@bb = global [256 x [256 x float]] zeroinitializer, align 64

;;  for (int nl = 0; nl < 10000000/256; nl++)
;;    for (int i = 0; i < 256; ++i)
;;      for (int j = 1; j < 256; j++)
;;        aa[j][i] = aa[j - 1][i] + bb[j][i];

; CHECK: Processing InnerLoopId = 2 and OuterLoopId = 1
; CHECK: Loops interchanged.

define float @s231() {
entry:
  br label %for.cond1.preheader

; Loop:
for.cond1.preheader:                              ; preds = %entry, %for.cond.cleanup3
  %nl.036 = phi i32 [ 0, %entry ], [ %inc23, %for.cond.cleanup3 ]
  br label %for.cond5.preheader

for.cond.cleanup3:                                ; preds = %for.cond.cleanup7
  %inc23 = add nuw nsw i32 %nl.036, 1
  %exitcond41 = icmp ne i32 %inc23, 39062
  br i1 %exitcond41, label %for.cond1.preheader, label %for.cond.cleanup

for.cond.cleanup7:                                ; preds = %for.body8
  %indvars.iv.next39 = add nuw nsw i64 %indvars.iv38, 1
  %exitcond40 = icmp ne i64 %indvars.iv.next39, 256
  br i1 %exitcond40, label %for.cond5.preheader, label %for.cond.cleanup3

for.body8:                                        ; preds = %for.cond5.preheader, %for.body8
  %indvars.iv = phi i64 [ 1, %for.cond5.preheader ], [ %indvars.iv.next, %for.body8 ]
  %0 = add nsw i64 %indvars.iv, -1
  %arrayidx10 = getelementptr inbounds [256 x [256 x float]], ptr @aa, i64 0, i64 %0, i64 %indvars.iv38
  %1 = load float, ptr %arrayidx10, align 4
  %arrayidx14 = getelementptr inbounds [256 x [256 x float]], ptr @bb, i64 0, i64 %indvars.iv, i64 %indvars.iv38
  %2 = load float, ptr %arrayidx14, align 4
  %add = fadd fast float %2, %1
  %arrayidx18 = getelementptr inbounds [256 x [256 x float]], ptr @aa, i64 0, i64 %indvars.iv, i64 %indvars.iv38
  store float %add, ptr %arrayidx18, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 256
  br i1 %exitcond, label %for.body8, label %for.cond.cleanup7

for.cond5.preheader:                              ; preds = %for.cond1.preheader, %for.cond.cleanup7
  %indvars.iv38 = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next39, %for.cond.cleanup7 ]
  br label %for.body8

; Exit blocks
for.cond.cleanup:                                 ; preds = %for.cond.cleanup3
  ret float undef
}
