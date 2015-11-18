; RUN: llc -mcpu=corei7 -mtriple=x86_64-linux < %s | FileCheck %s -check-prefix=CHECK

define void @foo() {
; Test that when determining the edge probability from a node in an inner loop
; to a node in an outer loop, the weights on edges in the inner loop should be
; ignored if we are building the chain for the outer loop.
;
; CHECK-LABEL: foo:
; CHECK: callq c
; CHECK: callq b

entry:
  %call = call zeroext i1 @a()
  br i1 %call, label %if.then, label %if.else, !prof !1

if.then:
  %call1 = call zeroext i1 @a()
  br i1 %call1, label %while.body, label %if.end.1, !prof !1

while.body:
  %call2 = call zeroext i1 @a()
  br i1 %call2, label %if.then.1, label %while.cond

if.then.1:
  call void @d()
  br label %while.cond

while.cond:
  %call3 = call zeroext i1 @a()
  br i1 %call3, label %while.body, label %if.end

if.end.1:
  call void @d()
  br label %if.end

if.else:
  call void @b()
  br label %if.end

if.end:
  call void @c()
  ret void
}

define void @bar() {
; Test that when determining the edge probability from a node in a loop to a
; node in its peer loop, the weights on edges in the first loop should be
; ignored.
;
; CHECK-LABEL: bar:
; CHECK: callq c
; CHECK: callq b

entry:
  %call = call zeroext i1 @a()
  br i1 %call, label %if.then, label %if.else, !prof !1

if.then:
  %call1 = call zeroext i1 @a()
  br i1 %call1, label %if.then, label %while.body, !prof !2

while.body:
  %call2 = call zeroext i1 @a()
  br i1 %call2, label %while.body, label %if.end, !prof !2

if.else:
  call void @b()
  br label %if.end

if.end:
  call void @c()
  ret void
}

define void @par() {
; Test that when determining the edge probability from a node in a loop to a
; node in its outer loop, the weights on edges in the outer loop should be
; ignored if we are building the chain for the inner loop.
;
; CHECK-LABEL: par:
; CHECK: callq c
; CHECK: callq d
; CHECK: callq b

entry:
  br label %if.cond

if.cond:
  %call = call zeroext i1 @a()
  br i1 %call, label %if.then, label %if.else, !prof !3

if.then:
  call void @b()
  br label %if.end

if.else:
  call void @c()
  %call1 = call zeroext i1 @a()
  br i1 %call1, label %if.end, label %exit, !prof !4

if.end:
  call void @d()
  %call2 = call zeroext i1 @a()
  br i1 %call2, label %if.cond, label %if.end.2, !prof !2

if.end.2:
  call void @e()
  br label %if.cond

exit:
  ret void
}

declare zeroext i1 @a()
declare void @b()
declare void @c()
declare void @d()
declare void @e()

!1 = !{!"branch_weights", i32 10, i32 1}
!2 = !{!"branch_weights", i32 100, i32 1}
!3 = !{!"branch_weights", i32 1, i32 100}
!4 = !{!"branch_weights", i32 1, i32 1}
