; RUN: llvm-as < %s | opt -analyze -scalar-evolution \
; RUN:   -scalar-evolution-max-iterations=0 | grep Unpredictable
; PR2088

define void @fun() {
entry:
        br label %loop
loop:
        %i = phi i8 [ 0, %entry ], [ %i.next, %loop ]
        %i.next = add i8 %i, 4
        %cond = icmp ne i8 %i.next, 6
        br i1 %cond, label %loop, label %exit
exit:
        ret void
}
