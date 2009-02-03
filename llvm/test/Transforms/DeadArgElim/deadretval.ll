; RUN: llvm-as < %s | opt -deadargelim | llvm-dis | not grep DEAD

; Dead arg only used by dead retval
define internal i32 @test(i32 %DEADARG) {
        ret i32 %DEADARG
}

define i32 @test2(i32 %A) {
        %DEAD = call i32 @test( i32 %A )                ; <i32> [#uses=0]
        ret i32 123
}

define i32 @test3() {
        %X = call i32 @test2( i32 3232 )                ; <i32> [#uses=1]
        %Y = add i32 %X, -123           ; <i32> [#uses=1]
        ret i32 %Y
}

