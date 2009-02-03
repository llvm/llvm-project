; RUN: llvm-as < %s | llc -march=alpha | not grep cmovlt
; RUN: llvm-as < %s | llc -march=alpha | grep cmoveq

define i64 @cmov_lt(i64 %a, i64 %c) {
entry:
        %tmp.1 = icmp slt i64 %c, 0             ; <i1> [#uses=1]
        %retval = select i1 %tmp.1, i64 %a, i64 10              ; <i64> [#uses=1]
        ret i64 %retval
}

define i64 @cmov_const(i64 %a, i64 %b, i64 %c) {
entry:
        %tmp.1 = icmp slt i64 %a, %b            ; <i1> [#uses=1]
        %retval = select i1 %tmp.1, i64 %c, i64 10              ; <i64> [#uses=1]
        ret i64 %retval
}

define i64 @cmov_lt2(i64 %a, i64 %c) {
entry:
        %tmp.1 = icmp sgt i64 %c, 0             ; <i1> [#uses=1]
        %retval = select i1 %tmp.1, i64 10, i64 %a              ; <i64> [#uses=1]
        ret i64 %retval
}
