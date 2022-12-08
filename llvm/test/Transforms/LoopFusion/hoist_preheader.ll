; RUN: opt -S -passes=loop-fusion < %s | FileCheck %s

define void @hoist_preheader(i32 %N) {

; CHECK:pre1:
; CHECK-NEXT:  %hoistme = add i32 1, %N
; CHECK-NEXT:  %hoistme2 = add i32 1, %hoistme
; CHECK-NEXT:  br label %body1
pre1:
  br label %body1

; CHECK: body1:
; CHECK-NOT:  %hoistme
body1:  ; preds = %pre1, %body1
  %i = phi i32 [%i_next, %body1], [0, %pre1]
  %i_next = add i32 1, %i
  %cond = icmp ne i32 %i, %N
  br i1 %cond, label %body1, label %pre2

pre2:
  %hoistme = add i32 1, %N
  %hoistme2 = add i32 1, %hoistme
  br label %body2

body2:  ; preds = %pre2, %body2
  %i2 = phi i32 [%i_next2, %body2], [0, %pre2]
  %i_next2 = add i32 1, %i2
  %cond2 = icmp ne i32 %i2, %N
  br i1 %cond2, label %body2, label %exit

exit:
  ret void
}
