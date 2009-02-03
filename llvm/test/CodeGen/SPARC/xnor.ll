; RUN: llvm-as < %s | llc -march=sparc | \
; RUN:   grep xnor | count 2

define i32 @test1(i32 %X, i32 %Y) {
        %A = xor i32 %X, %Y             ; <i32> [#uses=1]
        %B = xor i32 %A, -1             ; <i32> [#uses=1]
        ret i32 %B
}

define i32 @test2(i32 %X, i32 %Y) {
        %A = xor i32 %X, -1             ; <i32> [#uses=1]
        %B = xor i32 %A, %Y             ; <i32> [#uses=1]
        ret i32 %B
}

