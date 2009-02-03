; Make sure this testcase codegens to the ctpop instruction
; RUN: llvm-as < %s | llc -march=alpha -mcpu=ev67 | grep -i ctpop
; RUN: llvm-as < %s | llc -march=alpha -mattr=+CIX | \
; RUN:   grep -i ctpop
; RUN: llvm-as < %s | llc -march=alpha -mcpu=ev6 | \
; RUN:   not grep -i ctpop
; RUN: llvm-as < %s | llc -march=alpha -mattr=-CIX | \
; RUN:   not grep -i ctpop

declare i64 @llvm.ctpop.i64(i64)

define i64 @bar(i64 %x) {
entry:
        %tmp.1 = call i64 @llvm.ctpop.i64( i64 %x )             ; <i64> [#uses=1]
        ret i64 %tmp.1
}

