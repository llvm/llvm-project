; RUN: llvm-as < %s | llc -march=x86  -stats |& \
; RUN:   grep asm-printer | grep 7

define i32 @g(i32 %a, i32 %b) nounwind {
        %tmp.1 = shl i32 %b, 1          ; <i32> [#uses=1]
        %tmp.3 = add i32 %tmp.1, %a             ; <i32> [#uses=1]
        %tmp.5 = mul i32 %tmp.3, %a             ; <i32> [#uses=1]
        %tmp.8 = mul i32 %b, %b         ; <i32> [#uses=1]
        %tmp.9 = add i32 %tmp.5, %tmp.8         ; <i32> [#uses=1]
        ret i32 %tmp.9
}

