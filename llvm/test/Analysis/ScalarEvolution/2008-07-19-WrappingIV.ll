; RUN: opt < %s -analyze -scalar-evolution \
; RUN:   -scalar-evolution-max-iterations=0 | grep "backedge-taken count is 113"
; PR2088

define void @fun() {
entry:
        br label %loop
loop:
        %i = phi i8 [ 0, %entry ], [ %i.next, %loop ]
        %i.next = add i8 %i, 18
        %cond = icmp ne i8 %i.next, 4
        br i1 %cond, label %loop, label %exit
exit:
        ret void
}
