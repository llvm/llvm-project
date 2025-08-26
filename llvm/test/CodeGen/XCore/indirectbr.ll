; RUN: llc < %s -mtriple=xcore | FileCheck %s

@nextaddr = global ptr null                       ; <ptr> [#uses=2]
@C.0.2070 = private constant [5 x ptr] [ptr blockaddress(@foo, %L1), ptr blockaddress(@foo, %L2), ptr blockaddress(@foo, %L3), ptr blockaddress(@foo, %L4), ptr blockaddress(@foo, %L5)] ; <ptr> [#uses=1]

define internal i32 @foo(i32 %i) nounwind {
; CHECK-LABEL: foo:
entry:
  %0 = load ptr, ptr @nextaddr, align 4               ; <ptr> [#uses=2]
  %1 = icmp eq ptr %0, null                       ; <i1> [#uses=1]
  br i1 %1, label %bb3, label %bb2

bb2:                                              ; preds = %entry, %bb3
  %gotovar.4.0 = phi ptr [ %gotovar.4.0.pre, %bb3 ], [ %0, %entry ] ; <ptr> [#uses=1]
; CHECK: bau
  indirectbr ptr %gotovar.4.0, [label %L5, label %L4, label %L3, label %L2, label %L1]

bb3:                                              ; preds = %entry
  %2 = getelementptr inbounds [5 x ptr], ptr @C.0.2070, i32 0, i32 %i ; <ptr> [#uses=1]
  %gotovar.4.0.pre = load ptr, ptr %2, align 4        ; <ptr> [#uses=1]
  br label %bb2

L5:                                               ; preds = %bb2
  br label %L4

L4:                                               ; preds = %L5, %bb2
  %res.0 = phi i32 [ 385, %L5 ], [ 35, %bb2 ]     ; <i32> [#uses=1]
  br label %L3

L3:                                               ; preds = %L4, %bb2
  %res.1 = phi i32 [ %res.0, %L4 ], [ 5, %bb2 ]   ; <i32> [#uses=1]
  br label %L2

L2:                                               ; preds = %L3, %bb2
  %res.2 = phi i32 [ %res.1, %L3 ], [ 1, %bb2 ]   ; <i32> [#uses=1]
  %phitmp = mul i32 %res.2, 6                     ; <i32> [#uses=1]
  br label %L1

L1:                                               ; preds = %L2, %bb2
  %res.3 = phi i32 [ %phitmp, %L2 ], [ 2, %bb2 ]  ; <i32> [#uses=1]
; CHECK: ldap r11, .Ltmp0
; CHECK: stw r11, dp[nextaddr]
  store ptr blockaddress(@foo, %L5), ptr @nextaddr, align 4
  ret i32 %res.3
}
