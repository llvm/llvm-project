; RUN: opt < %s -passes=loop-vectorize -debug-only=loop-vectorize -S 2>&1 | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

define i32 @multi_user_cmp(ptr readonly %a, i32 noundef %n) {
; CHECK: LV: Found an estimated cost of 0 for VF 4 For instruction:   %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
; CHECK-NEXT: LV: Found an estimated cost of 0 for VF 4 For instruction:   %all.0.off010 = phi i1 [ true, %entry ], [ %all.0.off0., %for.body ]
; CHECK-NEXT: LV: Found an estimated cost of 0 for VF 4 For instruction:   %any.0.off09 = phi i1 [ false, %entry ], [ %.any.0.off0, %for.body ]
; CHECK-NEXT: LV: Found an estimated cost of 0 for VF 4 For instruction:   %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
; CHECK-NEXT: LV: Found an estimated cost of 1 for VF 4 For instruction:   %load1 = load float, ptr %arrayidx, align 4
; CHECK-NEXT: LV: Found an estimated cost of 1 for VF 4 For instruction:   %cmp1 = fcmp olt float %load1, 0.000000e+00
; CHECK-NEXT: LV: Found an estimated cost of 1 for VF 4 For instruction:   %.any.0.off0 = select i1 %cmp1, i1 true, i1 %any.0.off09
; CHECK-NEXT: LV: Found an estimated cost of 1 for VF 4 For instruction:   %all.0.off0. = select i1 %cmp1, i1 %all.0.off010, i1 false
; CHECK-NEXT: LV: Found an estimated cost of 1 for VF 4 For instruction:   %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK-NEXT: LV: Found an estimated cost of 1 for VF 4 For instruction:   %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
; CHECK-NEXT: LV: Found an estimated cost of 0 for VF 4 For instruction:   br i1 %exitcond.not, label %exit, label %for.body
entry:
  %wide.trip.count = zext nneg i32 %n to i64
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %all.0.off010 = phi i1 [ true, %entry ], [ %all.0.off0., %for.body ]
  %any.0.off09 = phi i1 [ false, %entry ], [ %.any.0.off0, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %load1 = load float, ptr %arrayidx, align 4
  %cmp1 = fcmp olt float %load1, 0.000000e+00
  %.any.0.off0 = select i1 %cmp1, i1 true, i1 %any.0.off09
  %all.0.off0. = select i1 %cmp1, i1 %all.0.off010, i1 false
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  %0 = select i1 %.any.0.off0, i32 2, i32 3
  %1 = select i1 %all.0.off0., i32 1, i32 %0
  ret i32 %1
}
