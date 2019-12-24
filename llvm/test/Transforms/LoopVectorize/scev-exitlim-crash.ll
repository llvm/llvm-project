; RUN: opt < %s -loop-vectorize -force-vector-interleave=2 -force-vector-width=8 -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

@b = common global i32 0, align 4
@f = common global i32 0, align 4
@a = common global i32 0, align 4
@d = common global i32* null, align 8
@e = common global i32* null, align 8
@c = common global i32 0, align 4

; CHECK-LABEL: @fn1(
; CHECK: vector.body
define void @fn1() #0 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %entry
  %i.0 = phi i32 [ undef, %entry ], [ %inc, %for.cond ]
  %cmp = icmp slt i32 %i.0, 0
  %call = tail call i32 @fn2(double fadd (double fsub (double undef, double undef), double 1.000000e+00)) #2
  %inc = add nsw i32 %i.0, 1
  br i1 %cmp, label %for.cond, label %for.cond4.preheader

for.cond4.preheader:                              ; preds = %for.cond
  %call.lcssa = phi i32 [ %call, %for.cond ]
  %cmp514 = icmp sgt i32 %call.lcssa, 0
  br i1 %cmp514, label %for.cond7.preheader.lr.ph, label %for.end26

for.cond7.preheader.lr.ph:                        ; preds = %for.cond4.preheader
  %0 = load i32*, i32** @e, align 8, !tbaa !4
  br label %for.cond7.preheader

for.cond7.preheader:                              ; preds = %for.cond7.preheader.lr.ph, %for.inc23
  %y.017 = phi i32 [ 0, %for.cond7.preheader.lr.ph ], [ %inc24, %for.inc23 ]
  %i.116 = phi i32 [ 0, %for.cond7.preheader.lr.ph ], [ %i.2.lcssa, %for.inc23 ]
  %n.015 = phi i32 [ undef, %for.cond7.preheader.lr.ph ], [ %inc25, %for.inc23 ]
  %1 = load i32, i32* @b, align 4, !tbaa !5
  %tobool11 = icmp eq i32 %1, 0
  br i1 %tobool11, label %for.inc23, label %for.body8.lr.ph

for.body8.lr.ph:                                  ; preds = %for.cond7.preheader
  %add9 = add i32 %n.015, 1
  br label %for.body8

for.body8:                                        ; preds = %for.body8.lr.ph, %for.inc19
  %indvars.iv19 = phi i64 [ 0, %for.body8.lr.ph ], [ %indvars.iv.next20, %for.inc19 ]
  %i.213 = phi i32 [ %i.116, %for.body8.lr.ph ], [ 0, %for.inc19 ]
  %2 = trunc i64 %indvars.iv19 to i32
  %add10 = add i32 %add9, %2
  store i32 %add10, i32* @f, align 4, !tbaa !5
  %idx.ext = sext i32 %add10 to i64
  %add.ptr = getelementptr inbounds i32, i32* @a, i64 %idx.ext
  %tobool129 = icmp eq i32 %i.213, 0
  br i1 %tobool129, label %for.inc19, label %for.body13.lr.ph

for.body13.lr.ph:                                 ; preds = %for.body8
  %3 = sext i32 %i.213 to i64
  br label %for.body13

for.body13:                                       ; preds = %for.body13.lr.ph, %for.body13
  %indvars.iv = phi i64 [ %3, %for.body13.lr.ph ], [ %indvars.iv.next, %for.body13 ]
  %add.ptr.sum = add i64 %idx.ext, %indvars.iv
  %arrayidx = getelementptr inbounds i32, i32* @a, i64 %add.ptr.sum
  %4 = load i32, i32* %arrayidx, align 4, !tbaa !5
  %arrayidx15 = getelementptr inbounds i32, i32* %0, i64 %indvars.iv
  store i32 %4, i32* %arrayidx15, align 4, !tbaa !5
  %indvars.iv.next = add i64 %indvars.iv, 1
  %5 = trunc i64 %indvars.iv.next to i32
  %tobool12 = icmp eq i32 %5, 0
  br i1 %tobool12, label %for.cond11.for.inc19_crit_edge, label %for.body13

for.cond11.for.inc19_crit_edge:                   ; preds = %for.body13
  br label %for.inc19

for.inc19:                                        ; preds = %for.cond11.for.inc19_crit_edge, %for.body8
  %6 = load i32, i32* @c, align 4, !tbaa !5
  %inc20 = add nsw i32 %6, 1
  store i32 %inc20, i32* @c, align 4, !tbaa !5
  %indvars.iv.next20 = add i64 %indvars.iv19, 1
  %7 = load i32, i32* @b, align 4, !tbaa !5
  %tobool = icmp eq i32 %7, 0
  br i1 %tobool, label %for.cond7.for.inc23_crit_edge, label %for.body8

for.cond7.for.inc23_crit_edge:                    ; preds = %for.inc19
  %add.ptr.lcssa = phi i32* [ %add.ptr, %for.inc19 ]
  store i32* %add.ptr.lcssa, i32** @d, align 8, !tbaa !4
  br label %for.inc23

for.inc23:                                        ; preds = %for.cond7.for.inc23_crit_edge, %for.cond7.preheader
  %i.2.lcssa = phi i32 [ 0, %for.cond7.for.inc23_crit_edge ], [ %i.116, %for.cond7.preheader ]
  %inc24 = add nsw i32 %y.017, 1
  %inc25 = add nsw i32 %n.015, 1
  %exitcond = icmp ne i32 %inc24, %call.lcssa
  br i1 %exitcond, label %for.cond7.preheader, label %for.cond4.for.end26_crit_edge

for.cond4.for.end26_crit_edge:                    ; preds = %for.inc23
  br label %for.end26

for.end26:                                        ; preds = %for.cond4.for.end26_crit_edge, %for.cond4.preheader
  ret void
}
declare i32 @fn2(double) #1

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

!0 = !{!"int", !1}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
!3 = !{!"double", !1}
!4 = !{!0, !0, i64 0}
!5 = !{!3, !3, i64 0}
