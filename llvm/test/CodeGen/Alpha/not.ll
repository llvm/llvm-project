; Make sure this testcase codegens to the ornot instruction
; RUN: llvm-as < %s | llc -march=alpha | grep eqv

define i64 @bar(i64 %x) {
entry:
        %tmp.1 = xor i64 %x, -1         ; <i64> [#uses=1]
        ret i64 %tmp.1
}
