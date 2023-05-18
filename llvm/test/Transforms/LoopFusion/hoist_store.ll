; RUN: opt -S -passes=loop-simplify,loop-fusion -debug-only=loop-fusion < %s 2>&1 | FileCheck %s
; REQUIRES: asserts
; CHECK: Safe to hoist.

@A = common global [100 x i32] zeroinitializer, align 16
define void @hoist_preheader(i32 %N) {
; CHECK-LABEL: @hoist_preheader(
; CHECK-NEXT:  pre1:
; CHECK-NEXT:    [[PTR:%.*]] = alloca i32, align 4
; CHECK-NEXT:    store i32 3, ptr [[PTR]], align 4
; CHECK-NEXT:    br label [[BODY1:%.*]]
; CHECK:       body1:
; CHECK-NEXT:    [[I:%.*]] = phi i32 [ [[I_NEXT:%.*]], [[BODY1]] ], [ 0, [[PRE1:%.*]] ]
; CHECK-NEXT:    [[I2:%.*]] = phi i32 [ [[I_NEXT2:%.*]], [[BODY1]] ], [ 0, [[PRE1]] ]
; CHECK-NEXT:    [[I_NEXT]] = add i32 1, [[I]]
; CHECK-NEXT:    [[COND:%.*]] = icmp ne i32 [[I]], [[N:%.*]]
; CHECK-NEXT:    [[I_NEXT2]] = add i32 1, [[I2]]
; CHECK-NEXT:    [[COND2:%.*]] = icmp ne i32 [[I2]], [[N]]
; CHECK-NEXT:    br i1 [[COND2]], label [[BODY1]], label [[EXIT:%.*]]
; CHECK:       exit:
; CHECK-NEXT:    ret void
;
pre1:
  %ptr = alloca i32
  br label %body1

body1:  ; preds = %pre1, %body1
  %i = phi i32 [%i_next, %body1], [0, %pre1]
  %i_next = add i32 1, %i
  %cond = icmp ne i32 %i, %N
  br i1 %cond, label %body1, label %pre2

pre2:
  store i32 3, ptr %ptr
  br label %body2

body2:  ; preds = %pre2, %body2
  %i2 = phi i32 [%i_next2, %body2], [0, %pre2]
  %i_next2 = add i32 1, %i2
  %cond2 = icmp ne i32 %i2, %N
  br i1 %cond2, label %body2, label %exit

exit:
  ret void
}
