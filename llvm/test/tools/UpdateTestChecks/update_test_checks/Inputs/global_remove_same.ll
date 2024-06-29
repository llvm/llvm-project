; RUN: opt -S < %s | FileCheck %s

define i32 @foo() {
; CHECK-LABEL: @foo(
; CHECK-NEXT:    [[RESULT:%.*]] = call i32 @bar(i32 0, i32 1)
; CHECK-NEXT:    ret i32 [[RESULT]]
;
  %result = call i32 @bar(i32 0, i32 1)
  ret i32 %result
}

declare i32 @bar(i32, i32)
; CHECK-LABEL: @bar(
; CHECK-SAME: i32
; CHECK-SAME: i32
