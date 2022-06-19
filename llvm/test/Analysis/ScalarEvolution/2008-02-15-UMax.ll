; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" 2>&1 | FileCheck %s
; PR2003

; CHECK: umax

define i32 @foo(i32 %n) {
entry:
        br label %header
header:
        %i = phi i32 [ 100, %entry ], [ %i.inc, %next ]
        %cond = icmp ult i32 %i, %n
        br i1 %cond, label %next, label %return
next:
        %i.inc = add i32 %i, 1
        br label %header
return:
        ret i32 %i
}

