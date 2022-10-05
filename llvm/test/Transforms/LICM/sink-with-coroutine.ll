; Verifies that LICM is disabled for loops that contains coro.suspend.
; RUN: opt -S < %s -passes=licm | FileCheck %s

define i64 @licm(i64 %n) #0 {
; CHECK-LABEL: @licm(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[P:%.*]] = alloca i64, align 8
; CHECK-NEXT:    br label [[BB0:%.*]]
; CHECK:       bb0:
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[I:%.*]] = phi i64 [ 0, [[BB0]] ], [ [[T5:%.*]], [[AWAIT_READY:%.*]] ]
; CHECK-NEXT:    [[T5]] = add i64 [[I]], 1
; CHECK-NEXT:    [[SUSPEND:%.*]] = call i8 @llvm.coro.suspend(token none, i1 false)
; CHECK-NEXT:    switch i8 [[SUSPEND]], label [[BB2:%.*]] [
; CHECK-NEXT:    i8 0, label [[AWAIT_READY]]
; CHECK-NEXT:    ]
; CHECK:       await.ready:
; CHECK-NEXT:    store i64 1, ptr [[P]], align 4
; CHECK-NEXT:    [[T6:%.*]] = icmp ult i64 [[T5]], [[N:%.*]]
; CHECK-NEXT:    br i1 [[T6]], label [[LOOP]], label [[BB2]]
; CHECK:       bb2:
; CHECK-NEXT:    [[RES:%.*]] = call i1 @llvm.coro.end(ptr null, i1 false)
; CHECK-NEXT:    ret i64 0
;
entry:
  %p = alloca i64
  br label %bb0

bb0:
  br label %loop

loop:
  %i = phi i64 [ 0, %bb0 ], [ %t5, %await.ready ]
  %t5 = add i64 %i, 1
  %suspend = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %suspend, label %bb2 [
  i8 0, label %await.ready
  ]

await.ready:
  store i64 1, ptr %p
  %t6 = icmp ult i64 %t5, %n
  br i1 %t6, label %loop, label %bb2

bb2:
  %res = call i1 @llvm.coro.end(ptr null, i1 false)
  ret i64 0
}

declare i8  @llvm.coro.suspend(token, i1)
declare i1  @llvm.coro.end(ptr, i1)
