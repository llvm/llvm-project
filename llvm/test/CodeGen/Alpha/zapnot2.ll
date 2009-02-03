; Make sure this testcase codegens to the zapnot instruction
; RUN: llvm-as < %s | llc -march=alpha | grep zapnot

define i64 @bar(i64 %x) {
entry:
        %tmp.1 = and i64 %x, 16711935           ; <i64> [#uses=1]
        ret i64 %tmp.1
}

