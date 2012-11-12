; RUN: llc < %s -march=ppc32 | grep extsb
; RUN: llc < %s -march=ppc32 | grep extsh

define i32 @p1(i8 %c, i16 %s) {
entry:
        %tmp = sext i8 %c to i32                ; <i32> [#uses=1]
        %tmp1 = sext i16 %s to i32              ; <i32> [#uses=1]
        %tmp2 = add i32 %tmp1, %tmp             ; <i32> [#uses=1]
        ret i32 %tmp2
}
