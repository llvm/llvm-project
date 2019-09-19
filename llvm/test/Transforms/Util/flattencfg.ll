; RUN: opt -flattencfg -S < %s | FileCheck %s


; This test checks whether the pass completes without a crash.
; The code is not transformed in any way
;
; CHECK-LABEL: @test_not_crash
define void @test_not_crash(i32 %in_a) #0 {
entry:
  %cmp0 = icmp eq i32 %in_a, -1
  %cmp1 = icmp ne i32 %in_a, 0
  %cond0 = and i1 %cmp0, %cmp1
  br i1 %cond0, label %b0, label %b1

b0:                                ; preds = %entry
  %cmp2 = icmp eq i32 %in_a, 0
  %cmp3 = icmp ne i32 %in_a, 1
  %cond1 = or i1 %cmp2, %cmp3
  br i1 %cond1, label %exit, label %b1

b1:                                       ; preds = %entry, %b0
  br label %exit

exit:                               ; preds = %entry, %b0, %b1
  ret void
}

; CHECK-LABEL: @test_not_crash2
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %0 = fcmp ult float %a
; CHECK-NEXT:    %1 = fcmp ult float %b
; CHECK-NEXT:    [[COND:%[a-z0-9]+]] = or i1 %0, %1
; CHECK-NEXT:    br i1 [[COND]], label %bb4, label %bb3
; CHECK:       bb3:
; CHECK-NEXT:    br label %bb4
; CHECK:       bb4:
; CHECK-NEXT:    ret void
define void @test_not_crash2(float %a, float %b) #0 {
entry:
  %0 = fcmp ult float %a, 1.000000e+00
  br i1 %0, label %bb0, label %bb1

bb3:                                               ; preds = %bb0
  br label %bb4

bb4:                                               ; preds = %bb0, %bb3
  ret void

bb1:                                               ; preds = %entry
  br label %bb0

bb0:                                               ; preds = %bb1, %entry
  %1 = fcmp ult float %b, 1.000000e+00
  br i1 %1, label %bb4, label %bb3
}

