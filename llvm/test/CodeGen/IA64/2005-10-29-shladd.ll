; this should turn into shladd 
; RUN: llvm-as < %s | llc -march=ia64 | grep shladd

define i64 @bogglesmoggle(i64 %X, i64 %Y) {
        %A = shl i64 %X, 3              ; <i64> [#uses=1]
        %B = add i64 %A, %Y             ; <i64> [#uses=1]
        ret i64 %B
}

