; RUN: opt -function-specialization -func-specialization-avg-iters-cost=3 -func-specialization-size-threshold=10 -S < %s | FileCheck %s

; CHECK-NOT: foo.{{[0-9]+}}

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

@A = external dso_local constant i32, align 4
@B = external dso_local constant i32, align 4
@C = external dso_local constant i32, align 4
@D = external dso_local constant i32, align 4

declare i1 @cond_begin()
declare i1 @cond_end()
declare i1 @getCond()

define internal i32 @foo(i32 %x, ptr %b, ptr %c) alwaysinline {
entry:
  br label %loop.entry

loop.entry:
  br label %loop2.entry

loop2.entry:
  br label %loop2.body

loop2.body:
  %0 = load i32, ptr %b, align 4
  %1 = load i32, ptr %c, align 4
  %add.0 = add nsw i32 %0, %1
  %add = add nsw i32 %add.0, %x
  br label %loop2.end

loop2.end:
  %cond.end = call i1 @cond_end()
  br i1 %cond.end, label %loop2.entry, label %loop.end

loop.end:
  %cond2.end = call i1 @getCond()
  br i1 %cond2.end, label %loop.entry, label %return

return:
  ret i32 %add
}

define dso_local i32 @bar(i32 %x, i32 %y) {
entry:
  %tobool = icmp ne i32 %x, 0
  br i1 %tobool, label %if.then, label %if.else

if.then:
  %call = call i32 @foo(i32 %x, ptr @A, ptr @C)
  br label %return

if.else:
  %call1 = call i32 @foo(i32 %y, ptr @B, ptr @D)
  br label %return

return:
  %retval.0 = phi i32 [ %call, %if.then ], [ %call1, %if.else ]
  ret i32 %retval.0
}