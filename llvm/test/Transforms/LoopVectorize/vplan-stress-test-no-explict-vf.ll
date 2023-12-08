; REQUIRES: asserts
; RUN: opt < %s  -S -passes=loop-vectorize -enable-vplan-native-path -vplan-build-stress-test -debug-only=loop-vectorize -disable-output 2>&1  | FileCheck %s

; This test checks that, when stress testing VPlan, if the computed VF
; is 1, we override it to VF = 4.

; CHECK: LV: VPlan computed VF 1.
; CHECK: LV: VPlan stress testing: overriding computed VF.
; CHECK: LV: Using VF 4 to build VPlans.
@arr2 = external global [8 x i32], align 16
@arr = external global [8 x [8 x i32]], align 16

; Function Attrs: norecurse nounwind uwtable
define void @foo(i32 %n) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.inc8, %entry
  %indvars.iv21 = phi i64 [ 0, %entry ], [ %indvars.iv.next22, %for.inc8 ]
  %arrayidx = getelementptr inbounds [8 x i32], ptr @arr2, i64 0, i64 %indvars.iv21
  %0 = trunc i64 %indvars.iv21 to i32
  store i32 %0, ptr %arrayidx, align 4
  %1 = trunc i64 %indvars.iv21 to i32
  %add = add nsw i32 %1, %n
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %for.body
  %indvars.iv = phi i64 [ 0, %for.body ], [ %indvars.iv.next, %for.body3 ]
  %arrayidx7 = getelementptr inbounds [8 x [8 x i32]], ptr @arr, i64 0, i64 %indvars.iv, i64 %indvars.iv21
  store i32 %add, ptr %arrayidx7, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 8
  br i1 %exitcond, label %for.inc8, label %for.body3

for.inc8:                                         ; preds = %for.body3
  %indvars.iv.next22 = add nuw nsw i64 %indvars.iv21, 1
  %exitcond23 = icmp eq i64 %indvars.iv.next22, 8
  br i1 %exitcond23, label %for.end10, label %for.body, !llvm.loop !1

for.end10:                                        ; preds = %for.inc8
  ret void
}

!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.vectorize.enable", i1 true}
