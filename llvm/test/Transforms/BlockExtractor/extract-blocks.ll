; RUN: echo 'foo bb9' > %t
; RUN: echo 'foo bb20' >> %t
; RUN: opt -S -passes=extract-blocks -extract-blocks-file=%t %s | FileCheck %s --check-prefix=CHECK-NO-ERASE
; RUN: opt -S -passes=extract-blocks -extract-blocks-file=%t -extract-blocks-erase-funcs %s | FileCheck %s --check-prefix=CHECK-ERASE

; CHECK-NO-ERASE: @foo(
; CHECK-NO-ERASE: @foo.bb9(
; CHECK-NO-ERASE: @foo.bb20(
; CHECK-ERASE: declare i32 @foo(
; CHECK-ERASE: @foo.bb9(
; CHECK-ERASE: @foo.bb20(
define i32 @foo(i32 %arg, i32 %arg1) {
bb:
  %tmp5 = icmp sgt i32 %arg, 0
  %tmp8 = icmp sgt i32 %arg1, 0
  %or.cond = and i1 %tmp5, %tmp8
  br i1 %or.cond, label %bb9, label %bb14

bb9:                                              ; preds = %bb
  %tmp12 = shl i32 %arg1, 2
  %tmp13 = add nsw i32 %tmp12, %arg
  br label %bb30

bb14:                                             ; preds = %bb
  %0 = and i32 %arg1, %arg
  %1 = icmp slt i32 %0, 0
  br i1 %1, label %bb20, label %bb26

bb20:                                             ; preds = %bb14
  %tmp22 = mul nsw i32 %arg, 3
  %tmp24 = sdiv i32 %arg1, 6
  %tmp25 = add nsw i32 %tmp24, %tmp22
  br label %bb30

bb26:                                             ; preds = %bb14
  %tmp29 = sub nsw i32 %arg, %arg1
  br label %bb30

bb30:                                             ; preds = %bb26, %bb20, %bb9
  %tmp.0 = phi i32 [ %tmp13, %bb9 ], [ %tmp25, %bb20 ], [ %tmp29, %bb26 ]
  ret i32 %tmp.0
}

