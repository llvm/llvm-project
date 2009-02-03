; RUN: llvm-as < %s | opt -analyze -scalar-evolution | grep -e {-->  %b}
; PR1810

define void @fun() {
entry:
        br label %header
header:
        %i = phi i32 [ 1, %entry ], [ %i.next, %body ]
        %cond = icmp eq i32 %i, 10
        br i1 %cond, label %exit, label %body
body:
        %a = mul i32 %i, 5
        %b = or i32 %a, 1
        %i.next = add i32 %i, 1
        br label %header
exit:        
        ret void
}
