; RUN: opt < %s -analyze -scalar-evolution | grep umax
; PR2003

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

