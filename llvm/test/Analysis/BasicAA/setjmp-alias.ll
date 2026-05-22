; Test for Issue #198967: DSE incorrectly removes store after setjmp
;
; This test verifies that when a function contains setjmp (returns_twice),
; stores to allocas are not incorrectly eliminated even when loaded pointers
; might point to them.
;
; The original bug: Clang -O2 optimized away `i = 13` because BasicAA said
; the store to %i and load via %p didn't alias. This is wrong because:
; 1. After setjmp, %p's value (loaded via volatile) may point to %i
; 2. The store may be observable through the pointer after longjmp
;
; RUN: opt -S -O2 %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@buf = global [1 x i64] zeroinitializer, align 16
@x = global i32 0, align 4
@ii = global i32 0, align 4

declare i32 @setjmp(ptr) returns_twice nounwind
declare void @redo()

define void @bar() {
; CHECK-LABEL: @bar
; CHECK: %sj = call i32 @setjmp
; CHECK: store i32 13, ptr %1
; CHECK: load volatile ptr, ptr %0
; CHECK: load i32, ptr
entry:
  %1 = alloca ptr, align 8
  %2 = alloca i32, align 4
  %sj = call i32 @setjmp(ptr @buf)
  %cond = load i32, ptr @x
  %cmp = icmp ne i32 %cond, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store volatile ptr %2, ptr %1
  call void @redo()
  br label %if.end

if.end:
  store i32 13, ptr %2
  %ptr = load volatile ptr, ptr %1
  %val = load i32, ptr %ptr
  store i32 %val, ptr @ii
  ret void
}