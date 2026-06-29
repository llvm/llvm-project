; RUN: llc -mtriple=arm64-apple-macosx < %s | FileCheck %s
; RUN: llc -mtriple=arm64-apple-macosx -global-isel < %s | FileCheck %s

; Oracle functions with no uses should be removed entirely.
; CHECK-NOT: oracle_unused1
; CHECK-NOT: oracle_unused2

; A normal function should not be affected regardless of uses.
; CHECK-LABEL: normal_unused:
; CHECK:       ret

define void @oracle_unused1(ptr %p, i64 %n) #0 {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %gep = getelementptr inbounds i32, ptr %p, i64 %i
  %val = load i32, ptr %gep, align 4
  %add = add nsw i32 %val, 42
  store i32 %add, ptr %gep, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define i64 @oracle_unused2(ptr %p, i64 %n) #0 {
entry:
  %cmp0 = icmp eq i64 %n, 0
  br i1 %cmp0, label %exit, label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop.latch ]
  %sum = phi i64 [ 0, %entry ], [ %sum.next, %loop.latch ]
  %gep = getelementptr inbounds i64, ptr %p, i64 %i
  %val = load i64, ptr %gep, align 8
  %sum.next = add i64 %sum, %val
  br label %loop.latch

loop.latch:
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp ult i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  %result = phi i64 [ 0, %entry ], [ %sum.next, %loop.latch ]
  ret i64 %result
}

define void @normal_unused() {
  ret void
}

attributes #0 = { "oracle-function" }
