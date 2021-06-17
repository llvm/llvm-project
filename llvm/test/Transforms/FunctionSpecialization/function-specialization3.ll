; RUN: opt -function-specialization -func-specialization-avg-iters-cost=3 -S < %s | \
; RUN:   FileCheck %s --check-prefixes=COMMON,DISABLED
; RUN: opt -function-specialization -force-function-specialization -S < %s | \
; RUN:   FileCheck %s --check-prefixes=COMMON,FORCE
; RUN: opt -function-specialization -func-specialization-avg-iters-cost=3 -force-function-specialization -S < %s | \
; RUN:   FileCheck %s --check-prefixes=COMMON,FORCE

; Test for specializing a constant global.

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

@A = external dso_local constant i32, align 4
@B = external dso_local constant i32, align 4

define dso_local i32 @bar(i32 %x, i32 %y) {
; COMMON-LABEL: @bar
; FORCE:        %call = call i32 @foo.2(i32 %x, i32* @A)
; FORCE:        %call1 = call i32 @foo.1(i32 %y, i32* @B)
; DISABLED-NOT: %call1 = call i32 @foo.1(
entry:
  %tobool = icmp ne i32 %x, 0
  br i1 %tobool, label %if.then, label %if.else

if.then:
  %call = call i32 @foo(i32 %x, i32* @A)
  br label %return

if.else:
  %call1 = call i32 @foo(i32 %y, i32* @B)
  br label %return

return:
  %retval.0 = phi i32 [ %call, %if.then ], [ %call1, %if.else ]
  ret i32 %retval.0
}

; FORCE:      define internal i32 @foo.1(i32 %x, i32* %b) {
; FORCE-NEXT: entry:
; FORCE-NEXT:   %0 = load i32, i32* @B, align 4
; FORCE-NEXT:   %add = add nsw i32 %x, %0
; FORCE-NEXT:   ret i32 %add
; FORCE-NEXT: }

; FORCE:      define internal i32 @foo.2(i32 %x, i32* %b) {
; FORCE-NEXT: entry:
; FORCE-NEXT:   %0 = load i32, i32* @A, align 4
; FORCE-NEXT:   %add = add nsw i32 %x, %0
; FORCE-NEXT:   ret i32 %add
; FORCE-NEXT: }

define internal i32 @foo(i32 %x, i32* %b) {
entry:
  %0 = load i32, i32* %b, align 4
  %add = add nsw i32 %x, %0
  ret i32 %add
}
