;RUN: opt -S -simplifycfg -mtriple=arm < %s | FileCheck %s
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

; CHECK-LABEL: select_trunc_i64
; CHECK-NOT: br
; CHECK: select
; CHECK: select
define arm_aapcscc i32 @select_trunc_i64(i32 %a, i32 %b) {
entry:
  %conv = sext i32 %a to i64
  %conv1 = sext i32 %b to i64
  %add = add nsw i64 %conv1, %conv
  %cmp = icmp sgt i64 %add, 2147483647
  br i1 %cmp, label %cond.end7, label %cond.false

cond.false:                                       ; preds = %entry
  %0 = icmp sgt i64 %add, -2147483648
  %cond = select i1 %0, i64 %add, i64 -2147483648
  %extract.t = trunc i64 %cond to i32
  br label %cond.end7

cond.end7:                                        ; preds = %cond.false, %entry
  %cond8.off0 = phi i32 [ 2147483647, %entry ], [ %extract.t, %cond.false ]
  ret i32 %cond8.off0
}
