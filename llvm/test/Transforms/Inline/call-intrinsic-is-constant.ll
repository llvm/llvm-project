; RUN: opt %s -passes=inline -inline-threshold=20 -S | FileCheck %s

declare i1 @llvm.is.constant.i64(i64)
declare void @foo()

define void @callee(i64 %val) {
  %cond = call i1 @llvm.is.constant.i64(i64 %val)
  br i1 %cond, label %cond.true, label %cond.false

cond.true:
; Rack up costs with a couple of function calls so that this function
; gets inlined only when @llvm.is.constant.i64 is folded.  In reality,
; the "then" clause of __builtin_constant_p tends to have statements
; that fold very well, so the cost of the "then" clause is not a huge
; concern.
  call void @foo()
  call void @foo()
  ret void

cond.false:
  ret void
}

define void @caller(i64 %val) {
; CHECK-LABEL: @caller(
; CHECK-NEXT:    [[COND_I:%.*]] = call i1 @llvm.is.constant.i64(i64 [[VAL:%.*]])
; CHECK-NEXT:    br i1 [[COND_I]], label %[[COND_TRUE_I:.*]], label %[[COND_FALSE_I:.*]]
; CHECK:       [[COND_TRUE_I]]:
; CHECK-NEXT:    call void @foo()
; CHECK-NEXT:    call void @foo()
; CHECK-NEXT:    br label %[[CALLEE_EXIT:.*]]
; CHECK:       [[COND_FALSE_I]]:
; CHECK-NEXT:    br label %[[CALLEE_EXIT]]
; CHECK:       [[CALLEE_EXIT]]:
; CHECK-NEXT:    ret void
;
  call void @callee(i64 %val)
  ret void
}
