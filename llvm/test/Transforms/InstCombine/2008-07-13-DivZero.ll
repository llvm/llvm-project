; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {lshr.*3}
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {call .*%cond}
; PR2506

; We can simplify the operand of udiv to '8', but not the operand to the
; call.  If the callee never returns, we can't assume the div is reachable.
define i32 @a(i32 %x, i32 %y) {
entry:
        %tobool = icmp ne i32 %y, 0             ; <i1> [#uses=1]
        %cond = select i1 %tobool, i32 8, i32 0         ; <i32> [#uses=2]
        %call = call i32 @b( i32 %cond )                ; <i32> [#uses=0]
        %div = udiv i32 %x, %cond               ; <i32> [#uses=1]
        ret i32 %div
}

declare i32 @b(i32)
