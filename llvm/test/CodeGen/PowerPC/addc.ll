; All of these should be codegen'd without loading immediates
; RUN: llvm-as < %s | llc -march=ppc32 -o %t -f
; RUN: grep addc %t | count 1
; RUN: grep adde %t | count 1
; RUN: grep addze %t | count 1
; RUN: grep addme %t | count 1
; RUN: grep addic %t | count 2

define i64 @add_ll(i64 %a, i64 %b) {
entry:
        %tmp.2 = add i64 %b, %a         ; <i64> [#uses=1]
        ret i64 %tmp.2
}

define i64 @add_l_5(i64 %a) {
entry:
        %tmp.1 = add i64 %a, 5          ; <i64> [#uses=1]
        ret i64 %tmp.1
}

define i64 @add_l_m5(i64 %a) {
entry:
        %tmp.1 = add i64 %a, -5         ; <i64> [#uses=1]
        ret i64 %tmp.1
}

