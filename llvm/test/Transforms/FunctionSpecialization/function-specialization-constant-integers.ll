; RUN: opt -function-specialization -function-specialization-for-literal-constant=true -func-specialization-size-threshold=10 -S < %s | FileCheck %s

; Check that the literal constant parameter could be specialized.
; CHECK: @foo.1(
; CHECK: @foo.2(

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

declare i32 @getValue()
declare i1 @getCond()

define internal i32 @foo(i1 %break_cond) {
entry:
  br label %loop.entry

loop.entry:
  br label %loop2.entry

loop2.entry:
  br label %loop2.body

loop2.body:
  %value = call i32 @getValue()
  br i1 %break_cond, label %loop2.end, label %return

loop2.end:
  %cond.end = call i1 @getCond()
  br i1 %cond.end, label %loop2.entry, label %loop.end

loop.end:
  %cond2.end = call i1 @getCond()
  br i1 %cond2.end, label %loop.entry, label %return

return:
  ret i32 %value
}

define dso_local i32 @bar(i32 %x, i32 %y) {
entry:
  %retval.1 = call i32 @foo(i1 1)
  %retval.2 = call i32 @foo(i1 0)
  %retval = add nsw i32 %retval.1, %retval.2
  ret i32 %retval
}