; RUN: echo FIRST-RUN
; RUN: echo SECOND-RUN
define i32 @foo(i32 %a) { ret i32 0 }
define i32 @bar() { ret i32 1 }
