; RUN: llvm-as < %s | llc -march=ppc32 | not grep rlwinm

define i32 @setcc_one_or_zero(i32* %a) {
entry:
        %tmp.1 = icmp ne i32* %a, null          ; <i1> [#uses=1]
        %inc.1 = zext i1 %tmp.1 to i32          ; <i32> [#uses=1]
        ret i32 %inc.1
}

