; RUN: opt -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S < %s | FileCheck %s

; Just checking for lack of crash here, but we should be able to check the IR?
; Earlier version using auto-generated checks from utils/update_test_checks.py
; had bot problems though...

define void @patatino() {

; CHECK-LABEL: @patatino

  br label %bb1
bb1:                                              ; preds = %bb36, %0
  br label %bb2
bb2:                                              ; preds = %bb3, %bb1
  br i1 undef, label %bb4, label %bb3
bb3:                                              ; preds = %bb4, %bb2
  br i1 undef, label %bb2, label %bb5
bb4:                                              ; preds = %bb2
  switch i32 undef, label %bb3 [
  ]
bb5:                                              ; preds = %bb3
  br label %bb6
bb6:                                              ; preds = %bb5
  br i1 undef, label %bb7, label %bb9
bb7:                                              ; preds = %bb6
  %tmp = or i64 undef, 1
  %tmp8 = icmp ult i64 %tmp, 0
  br i1 %tmp8, label %bb12, label %bb9
bb9:                                              ; preds = %bb35, %bb34, %bb33, %bb32, %bb31, %bb30, %bb27, %bb24, %bb21, %bb18, %bb16, %bb14, %bb12, %bb7, %bb6
  br label %bb11
bb10:                                             ; preds = %bb36
  br label %bb11
bb11:                                             ; preds = %bb10, %bb9
  ret void
bb12:                                             ; preds = %bb7
  %tmp13 = icmp ult i64 0, 0
  br i1 %tmp13, label %bb14, label %bb9
bb14:                                             ; preds = %bb12
  %tmp15 = icmp ult i64 undef, 0
  br i1 %tmp15, label %bb16, label %bb9
bb16:                                             ; preds = %bb14
  %tmp17 = icmp ult i64 undef, 0
  br i1 %tmp17, label %bb18, label %bb9
bb18:                                             ; preds = %bb16
  %tmp19 = or i64 undef, 5
  %tmp20 = icmp ult i64 %tmp19, 0
  br i1 %tmp20, label %bb21, label %bb9
bb21:                                             ; preds = %bb18
  %tmp22 = or i64 undef, 6
  %tmp23 = icmp ult i64 %tmp22, 0
  br i1 %tmp23, label %bb24, label %bb9
bb24:                                             ; preds = %bb21
  %tmp25 = or i64 undef, 7
  %tmp26 = icmp ult i64 %tmp25, 0
  br i1 %tmp26, label %bb27, label %bb9
bb27:                                             ; preds = %bb24
  %tmp28 = or i64 undef, 8
  %tmp29 = icmp ult i64 %tmp28, 0
  br i1 %tmp29, label %bb30, label %bb9
bb30:                                             ; preds = %bb27
  br i1 undef, label %bb31, label %bb9
bb31:                                             ; preds = %bb30
  br i1 undef, label %bb32, label %bb9
bb32:                                             ; preds = %bb31
  br i1 undef, label %bb33, label %bb9
bb33:                                             ; preds = %bb32
  br i1 undef, label %bb34, label %bb9
bb34:                                             ; preds = %bb33
  br i1 undef, label %bb35, label %bb9
bb35:                                             ; preds = %bb34
  br i1 undef, label %bb36, label %bb9
bb36:                                             ; preds = %bb35
  br i1 undef, label %bb1, label %bb10
}
