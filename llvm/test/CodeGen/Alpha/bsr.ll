; Make sure this testcase codegens the bsr instruction
; RUN: llvm-as < %s | llc -march=alpha | grep bsr

define internal i64 @abc(i32 %x) {
        %tmp.2 = add i32 %x, -1         ; <i32> [#uses=1]
        %tmp.0 = call i64 @abc( i32 %tmp.2 )            ; <i64> [#uses=1]
        %tmp.5 = add i32 %x, -2         ; <i32> [#uses=1]
        %tmp.3 = call i64 @abc( i32 %tmp.5 )            ; <i64> [#uses=1]
        %tmp.6 = add i64 %tmp.0, %tmp.3         ; <i64> [#uses=1]
        ret i64 %tmp.6
}

