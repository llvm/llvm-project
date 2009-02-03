; Make sure this testcase does not use ctpop
; RUN: llvm-as < %s | llc -march=alpha | not grep -i ctpop 

declare i64 @llvm.ctlz.i64(i64)

define i64 @bar(i64 %x) {
entry:
        %tmp.1 = call i64 @llvm.ctlz.i64( i64 %x )              ; <i64> [#uses=1]
        ret i64 %tmp.1
}

