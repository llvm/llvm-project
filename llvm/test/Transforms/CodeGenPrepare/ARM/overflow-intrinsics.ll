; RUN: opt -codegenprepare -S < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8m.main-arm-none-eabi"

; CHECK-LABEL: uadd_overflow_too_far_cmp_dom
; CHECK-NOT: with.overflow.i32
define i32 @uadd_overflow_too_far_cmp_dom(i32 %arg0) {
entry:
  %cmp = icmp ne i32 %arg0, 0
  br i1 %cmp, label %if.else, label %if.then

if.then:
  call void @foo()
  br label %exit

if.else:
  call void @bar()
  br label %if.end

if.end:
  %dec = add nsw i32 %arg0, -1
  br label %exit

exit:
  %res = phi i32 [ %arg0, %if.then ], [ %dec, %if.end ]
  ret i32 %res
}

; CHECK-LABEL: uadd_overflow_too_far_math_dom
; CHECK-NOT: with.overflow.i32
define i32 @uadd_overflow_too_far_math_dom(i32 %arg0, i32 %arg1) {
entry:
  %dec = add nsw i32 %arg0, -1
  %cmp = icmp ugt i32 %arg0, 1
  br i1 %cmp, label %if.else, label %if.then

if.then:
  call void @foo()
  br label %if.end

if.else:
  call void @bar()
  br label %if.end

if.end:
  %cmp.i.i = icmp ne i32 %arg0, 0
  %tobool = zext i1 %cmp.i.i to i32
  br label %exit

exit:
  ret i32 %tobool
}

declare void @foo()
declare void @bar()
