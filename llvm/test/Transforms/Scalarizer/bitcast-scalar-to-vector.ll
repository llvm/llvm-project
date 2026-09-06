; RUN: opt %s -passes=scalarizer -S --data-layout=e | FileCheck %s --check-prefix=LE
; RUN: opt %s -passes=scalarizer -S --data-layout=E | FileCheck %s --check-prefix=BE

define i32 @split_via_extract(i64 noundef %a) {
; LE-LABEL: define i32 @split_via_extract(
; LE-SAME: i64 noundef [[A:%.*]]) {
; LE: [[LO:%.*]] = trunc i64 [[A]] to i32
; LE: [[SHIFT:%.*]] = lshr i64 [[A]], 32
; LE: [[HI:%.*]] = trunc i64 [[SHIFT]] to i32
; LE: [[RESULT:%.*]] = add i32 [[LO]], [[HI]]
; LE-NOT: bitcast
; LE: ret i32 [[RESULT]]
;
; BE-LABEL: define i32 @split_via_extract(
; BE-SAME: i64 noundef [[A:%.*]]) {
; BE: [[SHIFT:%.*]] = lshr i64 [[A]], 32
; BE: [[HI:%.*]] = trunc i64 [[SHIFT]] to i32
; BE: [[LO:%.*]] = trunc i64 [[A]] to i32
; BE: [[RESULT:%.*]] = add i32 [[HI]], [[LO]]
; BE-NOT: bitcast
; BE: ret i32 [[RESULT]]
entry:
  %vecA = bitcast i64 %a to <2 x i32>
  %low = extractelement <2 x i32> %vecA, i32 0
  %high = extractelement <2 x i32> %vecA, i32 1
  %result = add i32 %low, %high
  ret i32 %result
}
