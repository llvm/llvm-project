; All of these ands and shifts should be folded into rlwimi's
; RUN: llvm-as < %s | llc -march=ppc32 -o %t -f
; RUN: not grep mulhwu %t
; RUN: not grep srawi %t 
; RUN: not grep add %t 
; RUN: grep mulhw %t | count 1

define i32 @mulhs(i32 %a, i32 %b) {
entry:
        %tmp.1 = sext i32 %a to i64             ; <i64> [#uses=1]
        %tmp.3 = sext i32 %b to i64             ; <i64> [#uses=1]
        %tmp.4 = mul i64 %tmp.3, %tmp.1         ; <i64> [#uses=1]
        %tmp.6 = lshr i64 %tmp.4, 32            ; <i64> [#uses=1]
        %tmp.7 = trunc i64 %tmp.6 to i32                ; <i32> [#uses=1]
        ret i32 %tmp.7
}

