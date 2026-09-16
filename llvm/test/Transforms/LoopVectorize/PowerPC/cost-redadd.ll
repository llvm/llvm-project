; REQUIRES: asserts
; RUN: opt -vectorizer-maximize-bandwidth -mcpu=pwr8 -S -passes=loop-vectorize -disable-output -debug-only=loop-vectorize < %s 2>&1 | FileCheck --check-prefix=P8COST %s
; NOTE: P9 uses 2x 64-bit vector units, P8 and P10 use 1x 128-bit vector unit
; RUN: opt -vectorizer-maximize-bandwidth -mcpu=pwr9 -S -passes=loop-vectorize -disable-output -debug-only=loop-vectorize < %s 2>&1 | FileCheck --check-prefix=P9COST %s
; RUN: opt -vectorizer-maximize-bandwidth -mcpu=pwr10 -S -passes=loop-vectorize -disable-output -debug-only=loop-vectorize < %s 2>&1 | FileCheck --check-prefix=P8COST %s

target datalayout = "e-m:e-Fn32-i64:64-i128:128-n32:64-S128-v256:256:256-v512:512:512"
target triple = "powerpc64le-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: read) uwtable
define signext i32 @adds8(ptr nofree noundef readonly captures(none) %a, i32 noundef signext %n) {
; P8COST,P9COST: LV: Checking a loop in 'adds8'
; P8COST: Cost of 1 for VF 16: EXPRESSION vp<%8> = ir<%sum.07> + partial.reduce.add (ir<%0> sext to i32)
; P8COST: Cost for VF 16: 4 (Estimated cost per lane: 0.25)
; P9COST: Cost of 2 for VF 16: EXPRESSION vp<%8> = ir<%sum.07> + partial.reduce.add (ir<%0> sext to i32)
; P9COST: Cost for VF 16: 6 (Estimated cost per lane: 0.375)
entry:
  %conv = zext nneg i32 %n to i64
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  %add.lcssa = phi i32 [ %add, %for.body ]
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %add.lcssa, %for.cond.cleanup.loopexit ]
  ret i32 %sum.0.lcssa

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.08 = phi i64 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %sum.07 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds nuw i8, ptr %a, i64 %i.08
  %0 = load i8, ptr %arrayidx, align 1, !tbaa !10
  %conv2 = sext i8 %0 to i32
  %add = add nsw i32 %sum.07, %conv2
  %inc = add nuw nsw i64 %i.08, 1
  %exitcond.not = icmp eq i64 %inc, %conv
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body, !llvm.loop !12
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: read) uwtable
define signext i32 @addu8(ptr nofree noundef readonly captures(none) %a, i32 noundef signext %n) {
; P8COST,P9COST: LV: Checking a loop in 'addu8'
; P8COST: Cost of 1 for VF 16: EXPRESSION vp<%8> = ir<%sum.07> + partial.reduce.add (ir<%0> zext to i32)
; P8COST: Cost for VF 16: 4 (Estimated cost per lane: 0.25)
; P9COST: Cost of 2 for VF 16: EXPRESSION vp<%8> = ir<%sum.07> + partial.reduce.add (ir<%0> zext to i32)
; P9COST: Cost for VF 16: 6 (Estimated cost per lane: 0.375)
entry:
  %conv = zext nneg i32 %n to i64
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %sum.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.08 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %sum.07 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw i8, ptr %a, i64 %i.08
  %0 = load i8, ptr %arrayidx, align 1, !tbaa !10
  %conv2 = zext i8 %0 to i32
  %add = add nuw nsw i32 %sum.07, %conv2
  %inc = add nuw nsw i64 %i.08, 1
  %exitcond.not = icmp eq i64 %inc, %conv
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !14
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: read) uwtable
define signext i32 @adds16(ptr nofree noundef readonly captures(none) %a, i32 noundef signext %n) {
; P8COST,P9COST: LV: Checking a loop in 'adds16'
; P8COST: Cost of 1 for VF 8: EXPRESSION vp<%8> = ir<%sum.07> + partial.reduce.add (ir<%0> sext to i32)
; P8COST: Cost for VF 8: 4 (Estimated cost per lane: 0.5)
; P9COST: Cost of 2 for VF 8: EXPRESSION vp<%8> = ir<%sum.07> + partial.reduce.add (ir<%0> sext to i32)
; P9COST: Cost for VF 8: 6 (Estimated cost per lane: 0.75)
entry:
  %conv = zext nneg i32 %n to i64
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %sum.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.08 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %sum.07 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw [2 x i8], ptr %a, i64 %i.08
  %0 = load i16, ptr %arrayidx, align 2, !tbaa !19
  %conv2 = sext i16 %0 to i32
  %add = add nsw i32 %sum.07, %conv2
  %inc = add nuw nsw i64 %i.08, 1
  %exitcond.not = icmp eq i64 %inc, %conv
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !20
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: read) uwtable
define signext i32 @addu16(ptr nofree noundef readonly captures(none) %a, i32 noundef signext %n) {
; P8COST,P9COST: LV: Checking a loop in 'addu16'
; P8COST: Cost of 1 for VF 8: EXPRESSION vp<%8> = ir<%sum.07> + partial.reduce.add (ir<%0> zext to i32)
; P8COST: Cost for VF 8: 4 (Estimated cost per lane: 0.5)
; P9COST: Cost of 2 for VF 8: EXPRESSION vp<%8> = ir<%sum.07> + partial.reduce.add (ir<%0> zext to i32)
; P9COST: Cost for VF 8: 6 (Estimated cost per lane: 0.75)
entry:
  %conv = zext nneg i32 %n to i64
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %sum.0.lcssa

for.body:                                         ; preds = %entry, %for.body
  %i.08 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %sum.07 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw [2 x i8], ptr %a, i64 %i.08
  %0 = load i16, ptr %arrayidx, align 2, !tbaa !19
  %conv2 = zext i16 %0 to i32
  %add = add nuw nsw i32 %sum.07, %conv2
  %inc = add nuw nsw i64 %i.08, 1
  %exitcond.not = icmp eq i64 %inc, %conv
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !22
}

!0 = !{!"Simple C/C++ TBAA"}
!1 = !{!"omnipotent char", !0, i64 0}
!5 = !{!"int", !1, i64 0}
!10 = !{!1, !1, i64 0}
!11 = !{!"llvm.loop.mustprogress"}
!12 = distinct !{!12, !11}
!14 = distinct !{!14, !11}
!15 = !{!"short", !1, i64 0}
!19 = !{!15, !15, i64 0}
!20 = distinct !{!20, !11}
!22 = distinct !{!22, !11}
