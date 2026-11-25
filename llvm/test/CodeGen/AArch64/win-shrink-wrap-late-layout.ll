; RUN: llc -mtriple=aarch64-pc-windows-msvc -O3 -verify-machineinstrs \
; RUN:   -filetype=obj %s -o %t.obj
; RUN: llvm-readobj --unwind %t.obj | FileCheck %s

@condition = internal global i32 0, align 4
@counter = internal global ptr null, align 8
@otherwise = internal global i32 0, align 4
@count = internal global i32 0, align 4
@value = internal global i8 0
@increment = internal global i32 0, align 4
@total = internal global i32 0, align 4

; Both sides of the diamond can fall through to the shrink-wrapped prologue.
; Block placement must not duplicate it into both predecessors.
define void @keep_single_prologue() nounwind uwtable {
; CHECK-LABEL: Function: keep_single_prologue +
; CHECK: Prologue [
; CHECK: sub sp, sp, #16
entry:
  %buf = alloca [7 x i8], align 1
  %cond = load i32, ptr @condition
  %skip = icmp eq i32 %cond, 0
  br i1 %skip, label %zero, label %bump

bump:
  %ptr = load ptr, ptr @counter
  %old = load i32, ptr %ptr
  %next = add i32 %old, 1
  store i32 %next, ptr %ptr
  br label %join

zero:
  store i32 0, ptr @otherwise
  br label %join

join:
  %n = load i32, ptr @count
  %empty = icmp eq i32 %n, 0
  br i1 %empty, label %exit, label %prepare

prepare:
  %byte = load i8, ptr @value
  %slot = getelementptr inbounds i8, ptr %buf, i64 1
  store i8 %byte, ptr %slot
  %initial = load i32, ptr @total
  %inc = load i32, ptr @increment
  %limit = zext i32 %n to i64
  br label %loop

loop:
  %i = phi i64 [ 0, %prepare ], [ %i.next, %latch ]
  %sum = phi i32 [ %initial, %prepare ], [ %sum.next, %latch ]
  %element = getelementptr inbounds i8, ptr %buf, i64 %i
  %loaded = load i8, ptr %element
  %iszero = icmp eq i8 %loaded, 0
  br i1 %iszero, label %latch, label %accumulate

accumulate:
  %added = add i32 %sum, %inc
  store i32 %added, ptr @total
  br label %latch

latch:
  %sum.next = phi i32 [ %sum, %loop ], [ %added, %accumulate ]
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %limit
  br i1 %done, label %exit, label %loop

exit:
  ret void
}
