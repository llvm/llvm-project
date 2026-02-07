; RUN: opt -S -passes=simplifycfg < %s | FileCheck %s
; RUN: opt -S -passes='simplifycfg' < %s | FileCheck %s

; CHECK-LABEL: @fold_load
; CHECK-NOT: bb1
; CHECK: load i32
define i32 @fold_load(i1 %cond, ptr %p) {
entry:
  br i1 %cond, label %pred, label %other

pred:
  %d = add i32 1, 1
  br label %bb1

other:
  %d2 = add i32 2, 2
  br label %exit

bb1:
  %x = load i32, ptr %p
  br label %exit

exit:
  %phi = phi i32 [ %x, %bb1 ], [ 42, %other ]
  ret i32 %phi
}

; CHECK-LABEL: @fold_arith
; CHECK-NOT: bb1
; CHECK: add i32 %a, 1
define i32 @fold_arith(i32 %a, i1 %cond) {
entry:
  br i1 %cond, label %pred, label %other

pred:
  %d = add i32 3, 3
  br label %bb1

other:
  %d2 = add i32 4, 4
  br label %exit

bb1:
  %x = add i32 %a, 1
  br label %exit

exit:
  %phi = phi i32 [ %x, %bb1 ], [ %a, %other ]
  ret i32 %phi
}

; CHECK-LABEL: @no_fold_multi_store
; CHECK: bb1:
; CHECK: store i32
; CHECK: ret void
define void @no_fold_multi_store(ptr %p, i1 %cond) {
entry:
  br i1 %cond, label %A, label %B

A:
  %dA = add i32 10, 1
  br label %bb1

B:
  %dB = add i32 20, 1
  br label %bb1

bb1:
  store i32 10, ptr %p
  br label %exit

exit:
  ret void
}
