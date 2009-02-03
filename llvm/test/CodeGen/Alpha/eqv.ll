; Make sure this testcase codegens to the eqv instruction
; RUN: llvm-as < %s | llc -march=alpha | grep eqv

define i64 @bar(i64 %x, i64 %y) {
entry:
        %tmp.1 = xor i64 %x, -1         ; <i64> [#uses=1]
        %tmp.2 = xor i64 %y, %tmp.1             ; <i64> [#uses=1]
        ret i64 %tmp.2
}

