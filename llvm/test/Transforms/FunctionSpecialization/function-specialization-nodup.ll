; RUN: opt -passes="ipsccp<func-spec>" -force-specialization -S < %s | FileCheck %s

; Function @foo has function attribute 'noduplicate', so check that we don't
; specialize it:

; CHECK-NOT: @foo.1(
; CHECK-NOT: @foo.2(

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

@A = external dso_local constant i32, align 4
@B = external dso_local constant i32, align 4

define dso_local i32 @bar(i32 %x, i32 %y) {
entry:
  %tobool = icmp ne i32 %x, 0
  br i1 %tobool, label %if.then, label %if.else

if.then:
  %call = call i32 @foo(i32 %x, ptr @A)
  br label %return

if.else:
  %call1 = call i32 @foo(i32 %y, ptr @B)
  br label %return

return:
  %retval.0 = phi i32 [ %call, %if.then ], [ %call1, %if.else ]
  ret i32 %retval.0
}

define internal i32 @foo(i32 %x, ptr %b) #0 {
entry:
  %0 = load i32, ptr %b, align 4
  %add = add nsw i32 %x, %0
  ret i32 %add
}

attributes #0 = { noduplicate }
