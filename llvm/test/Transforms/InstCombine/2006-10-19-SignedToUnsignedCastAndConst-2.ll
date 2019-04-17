; The optimizer should be able to remove cast operation here.
; RUN: opt < %s -instcombine -S | \
; RUN:    not grep sext.*i32

define i1 @eq_signed_to_small_unsigned(i8 %SB) {
        %Y = sext i8 %SB to i32         ; <i32> [#uses=1]
        %C = icmp eq i32 %Y, 17         ; <i1> [#uses=1]
        ret i1 %C
}

