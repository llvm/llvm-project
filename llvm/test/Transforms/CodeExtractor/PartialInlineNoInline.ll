; RUN: opt < %s -passes=partial-inliner -S -stats -pass-remarks=partial-inlining 2>&1 | FileCheck %s

@stat = external global i32, align 4

define i32 @inline_fail(i32 %count, ...) {
entry:
  %vargs = alloca ptr, align 8
  call void @llvm.va_start(ptr %vargs)
  %stat1 = load i32, ptr @stat, align 4
  %cmp = icmp slt i32 %stat1, 0
  br i1 %cmp, label %bb2, label %bb1

bb1:                                              ; preds = %entry
  %vg1 = add nsw i32 %stat1, 1
  store i32 %vg1, ptr @stat, align 4
  %va1 = va_arg ptr %vargs, i32
  call void @foo(i32 %count, i32 %va1) #2
  br label %bb2

bb2:                                              ; preds = %bb1, %entry
  %res = phi i32 [ 1, %bb1 ], [ 0, %entry ]
  call void @llvm.va_end(ptr %vargs)
  ret i32 %res
}

define i32 @caller(i32 %arg) {
bb:
  %res = tail call i32 (i32, ...) @inline_fail(i32 %arg, i32 %arg)
  ret i32 %res
}

declare void @foo(i32, i32)
declare void @llvm.va_start(ptr)
declare void @llvm.va_end(ptr)

; Check that no remarks have been emitted, inline_fail has not been partial
; inlined, no code has been extracted and the partial-inlining counter
; has not been incremented.

; CHECK-NOT: remark
; CHECK: tail call i32 (i32, ...) @inline_fail(i32 %arg, i32 %arg)
; CHECK-NOT: inline_fail.1_bb1
; CHECK-NOT: partial-inlining
