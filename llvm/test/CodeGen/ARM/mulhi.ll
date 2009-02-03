; RUN: llvm-as < %s | llc -march=arm -mattr=+v6
; RUN: llvm-as < %s | llc -march=arm -mattr=+v6 | \
; RUN:   grep smmul | count 1
; RUN: llvm-as < %s | llc -march=arm | grep umull | count 1

define i32 @smulhi(i32 %x, i32 %y) {
        %tmp = sext i32 %x to i64               ; <i64> [#uses=1]
        %tmp1 = sext i32 %y to i64              ; <i64> [#uses=1]
        %tmp2 = mul i64 %tmp1, %tmp             ; <i64> [#uses=1]
        %tmp3 = lshr i64 %tmp2, 32              ; <i64> [#uses=1]
        %tmp3.upgrd.1 = trunc i64 %tmp3 to i32          ; <i32> [#uses=1]
        ret i32 %tmp3.upgrd.1
}

define i32 @umulhi(i32 %x, i32 %y) {
        %tmp = zext i32 %x to i64               ; <i64> [#uses=1]
        %tmp1 = zext i32 %y to i64              ; <i64> [#uses=1]
        %tmp2 = mul i64 %tmp1, %tmp             ; <i64> [#uses=1]
        %tmp3 = lshr i64 %tmp2, 32              ; <i64> [#uses=1]
        %tmp3.upgrd.2 = trunc i64 %tmp3 to i32          ; <i32> [#uses=1]
        ret i32 %tmp3.upgrd.2
}
