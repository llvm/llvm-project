; Make sure this testcase codegens to the zapnot instruction
; RUN: llvm-as < %s | llc -march=alpha | grep zapnot

define i64 @foo(i64 %y) {
entry:
        %tmp = lshr i64 %y, 3           ; <i64> [#uses=1]
        %tmp2 = and i64 %tmp, 8191              ; <i64> [#uses=1]
        ret i64 %tmp2
}

