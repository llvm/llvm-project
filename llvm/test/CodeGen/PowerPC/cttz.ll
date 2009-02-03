; Make sure this testcase does not use ctpop
; RUN: llvm-as < %s | llc -march=ppc32 | grep -i cntlzw

declare i32 @llvm.cttz.i32(i32)

define i32 @bar(i32 %x) {
entry:
        %tmp.1 = call i32 @llvm.cttz.i32( i32 %x )              ; <i32> [#uses=1]
        ret i32 %tmp.1
}

