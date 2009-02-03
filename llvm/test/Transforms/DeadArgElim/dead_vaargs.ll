; RUN: llvm-as < %s | opt -deadargelim | llvm-dis | not grep 47 
; RUN: llvm-as < %s | opt -deadargelim | llvm-dis | not grep 1.0

define i32 @bar(i32 %A) {
        %tmp4 = tail call i32 (i32, ...)* @foo( i32 %A, i32 %A, i32 %A, i32 %A, i64 47, double 1.000000e+00 )   ; <i32> [#uses=1]
        ret i32 %tmp4
}

define internal i32 @foo(i32 %X, ...) {
        ret i32 %X
}

