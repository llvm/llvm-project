; RUN: llvm-as < %s | llc -march=alpha | grep rpcc

declare i64 @llvm.readcyclecounter()

define i64 @foo() {
entry:
        %tmp.1 = call i64 @llvm.readcyclecounter( )             ; <i64> [#uses=1]
        ret i64 %tmp.1
}
