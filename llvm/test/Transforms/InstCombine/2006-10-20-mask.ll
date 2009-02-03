; RUN: llvm-as %s -o - | opt -instcombine | llvm-dis | \
; RUN:    grep and

define i64 @foo(i64 %tmp, i64 %tmp2) {
        %tmp.upgrd.1 = trunc i64 %tmp to i32            ; <i32> [#uses=1]
        %tmp2.upgrd.2 = trunc i64 %tmp2 to i32          ; <i32> [#uses=1]
        %tmp3 = and i32 %tmp.upgrd.1, %tmp2.upgrd.2             ; <i32> [#uses=1]
        %tmp4 = zext i32 %tmp3 to i64           ; <i64> [#uses=1]
        ret i64 %tmp4
}

