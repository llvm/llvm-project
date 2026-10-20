; REQUIRES: asserts
; RUN: opt -vectorizer-maximize-bandwidth -mcpu=pwr8 -S -passes=loop-vectorize -disable-output -debug-only=loop-vectorize < %s 2>&1 | FileCheck --check-prefix=P8COST %s
; NOTE: P9 uses 2x 64-bit vector units, P8 and P10 use 1x 128-bit vector unit
; RUN: opt -vectorizer-maximize-bandwidth -mcpu=pwr9 -S -passes=loop-vectorize -disable-output -debug-only=loop-vectorize < %s 2>&1 | FileCheck --check-prefix=P9COST %s
; RUN: opt -vectorizer-maximize-bandwidth -mcpu=pwr10 -S -passes=loop-vectorize -disable-output -debug-only=loop-vectorize < %s 2>&1 | FileCheck --check-prefix=P8COST %s

target datalayout = "e-m:e-Fn32-i64:64-i128:128-n32:64-S128-v256:256:256-v512:512:512"
target triple = "powerpc64le-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: read) uwtable
define signext i32 @dotu8(ptr noundef readonly captures(none) %a, ptr noundef readonly captures(none) %b, i64 noundef %n) {
; P8COST,P9COST: LV: Checking a loop in 'dotu8'
; P8COST: Cost of 1 for VF 16: EXPRESSION vp<%8> = ir<%sum.08> + partial.reduce.add (mul nuw nsw (ir<%1> zext to i32), (ir<%0> zext to i32))
; P8COST: Cost for VF 16: 5 (Estimated cost per lane: 0.313)
; P9COST: Cost of 2 for VF 16: EXPRESSION vp<%8> = ir<%sum.08> + partial.reduce.add (mul nuw nsw (ir<%1> zext to i32), (ir<%0> zext to i32))
; P9COST: Cost for VF 16: 8 (Estimated cost per lane: 0.5)
entry:
  %cmp7 = icmp sgt i64 %n, 0
  br i1 %cmp7, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  %add.lcssa = phi i32 [ %add, %for.body ]
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %add.lcssa, %for.cond.cleanup.loopexit ]
  ret i32 %sum.0.lcssa

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.09 = phi i64 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %sum.08 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds nuw i8, ptr %a, i64 %i.09
  %0 = load i8, ptr %arrayidx, align 1, !tbaa !8
  %conv = zext i8 %0 to i32
  %arrayidx1 = getelementptr inbounds nuw i8, ptr %b, i64 %i.09
  %1 = load i8, ptr %arrayidx1, align 1, !tbaa !8
  %conv2 = zext i8 %1 to i32
  %mul = mul nuw nsw i32 %conv2, %conv
  %add = add nuw nsw i32 %mul, %sum.08
  %inc = add nuw nsw i64 %i.09, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body, !llvm.loop !9
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: read) uwtable
define signext i32 @dots8(ptr noundef readonly captures(none) %a, ptr noundef readonly captures(none) %b, i64 noundef %n) {
; P8COST,P9COST: LV: Checking a loop in 'dots8'
; P8COST: Cost for VF 16: 26 (Estimated cost per lane: 1.63)
; P9COST: Cost for VF 16: 36 (Estimated cost per lane: 2.25)
entry:
  %cmp7 = icmp sgt i64 %n, 0
  br i1 %cmp7, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %sum.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.09 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %sum.08 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw i8, ptr %a, i64 %i.09
  %0 = load i8, ptr %arrayidx, align 1, !tbaa !8
  %conv = sext i8 %0 to i32
  %arrayidx1 = getelementptr inbounds nuw i8, ptr %b, i64 %i.09
  %1 = load i8, ptr %arrayidx1, align 1, !tbaa !8
  %conv2 = sext i8 %1 to i32
  %mul = mul nsw i32 %conv2, %conv
  %add = add nsw i32 %mul, %sum.08
  %inc = add nuw nsw i64 %i.09, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !11
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: read) uwtable
define signext i32 @dotus8(ptr nofree noundef readonly captures(none) %a, ptr nofree noundef readonly captures(none) %b, i64 noundef %n) {
; P8COST,P9COST: LV: Checking a loop in 'dotus8'
; P8COST: Cost of 1 for VF 16: EXPRESSION vp<%8> = ir<%sum.08> + partial.reduce.add (mul nsw (ir<%1> sext to i32), (ir<%0> zext to i32))
; P9COST: Cost of 2 for VF 16: EXPRESSION vp<%8> = ir<%sum.08> + partial.reduce.add (mul nsw (ir<%1> sext to i32), (ir<%0> zext to i32))
entry:
  %cmp7 = icmp sgt i64 %n, 0
  br i1 %cmp7, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %sum.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.09 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %sum.08 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw i8, ptr %a, i64 %i.09
  %0 = load i8, ptr %arrayidx, align 1, !tbaa !8
  %conv = zext i8 %0 to i32
  %arrayidx1 = getelementptr inbounds nuw i8, ptr %b, i64 %i.09
  %1 = load i8, ptr %arrayidx1, align 1, !tbaa !8
  %conv2 = sext i8 %1 to i32
  %mul = mul nsw i32 %conv2, %conv
  %add = add nsw i32 %mul, %sum.08
  %inc = add nuw nsw i64 %i.09, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !13
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: read) uwtable
define signext range(i32 0, -2147483648) i32 @dotu16(ptr noundef readonly captures(none) %a, ptr noundef readonly captures(none) %b, i64 noundef %n) {
; P8COST,P9COST: LV: Checking a loop in 'dotu16'
; P8COST: Cost of 1 for VF 8: EXPRESSION vp<%8> = ir<%sum.08> + partial.reduce.add (mul nuw nsw (ir<%1> zext to i32), (ir<%0> zext to i32))
; P8COST: Cost for VF 8: 5 (Estimated cost per lane: 0.625)
; P9COST: Cost of 2 for VF 8: EXPRESSION vp<%8> = ir<%sum.08> + partial.reduce.add (mul nuw nsw (ir<%1> zext to i32), (ir<%0> zext to i32))
; P9COST: Cost for VF 8: 8 (Estimated cost per lane: 1)
entry:
  %cmp7 = icmp sgt i64 %n, 0
  br i1 %cmp7, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %sum.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.09 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %sum.08 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw [2 x i8], ptr %a, i64 %i.09
  %0 = load i16, ptr %arrayidx, align 2, !tbaa !12
  %conv = zext i16 %0 to i32
  %arrayidx1 = getelementptr inbounds nuw [2 x i8], ptr %b, i64 %i.09
  %1 = load i16, ptr %arrayidx1, align 2, !tbaa !12
  %conv2 = zext i16 %1 to i32
  %mul = mul nuw nsw i32 %conv2, %conv
  %add = add nuw nsw i32 %mul, %sum.08
  %inc = add nuw nsw i64 %i.09, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !14
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: read) uwtable
define signext i32 @dots16(ptr noundef readonly captures(none) %a, ptr noundef readonly captures(none) %b, i64 noundef %n) {
; P8COST,P9COST: LV: Checking a loop in 'dots16'
; P8COST: Cost of 1 for VF 8: EXPRESSION vp<%8> = ir<%sum.08> + partial.reduce.add (mul nsw (ir<%1> sext to i32), (ir<%0> sext to i32))
; P8COST: Cost for VF 8: 5 (Estimated cost per lane: 0.625)
; P9COST: Cost of 2 for VF 8: EXPRESSION vp<%8> = ir<%sum.08> + partial.reduce.add (mul nsw (ir<%1> sext to i32), (ir<%0> sext to i32))
; P9COST: Cost for VF 8: 8 (Estimated cost per lane: 1)
entry:
  %cmp7 = icmp sgt i64 %n, 0
  br i1 %cmp7, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %sum.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.09 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %sum.08 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw [2 x i8], ptr %a, i64 %i.09
  %0 = load i16, ptr %arrayidx, align 2, !tbaa !12
  %conv = sext i16 %0 to i32
  %arrayidx1 = getelementptr inbounds nuw [2 x i8], ptr %b, i64 %i.09
  %1 = load i16, ptr %arrayidx1, align 2, !tbaa !12
  %conv2 = sext i16 %1 to i32
  %mul = mul nsw i32 %conv2, %conv
  %add = add nsw i32 %mul, %sum.08
  %inc = add nuw nsw i64 %i.09, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !15
}

!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = !{!6, !6, i64 0}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.mustprogress"}
!11 = distinct !{!11, !10}
!12 = !{!13, !13, i64 0}
!13 = !{!"short", !6, i64 0}
!14 = distinct !{!14, !10}
!15 = distinct !{!15, !10}
;.
