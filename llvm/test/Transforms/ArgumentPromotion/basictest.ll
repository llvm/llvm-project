; RUN: opt < %s -basicaa -argpromotion -mem2reg -S | not grep alloca
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"
define internal i32 @test(i32* %X, i32* %Y) {
        %A = load i32* %X               ; <i32> [#uses=1]
        %B = load i32* %Y               ; <i32> [#uses=1]
        %C = add i32 %A, %B             ; <i32> [#uses=1]
        ret i32 %C
}

define internal i32 @caller(i32* %B) {
        %A = alloca i32         ; <i32*> [#uses=2]
        store i32 1, i32* %A
        %C = call i32 @test( i32* %A, i32* %B )         ; <i32> [#uses=1]
        ret i32 %C
}

define i32 @callercaller() {
        %B = alloca i32         ; <i32*> [#uses=2]
        store i32 2, i32* %B
        %X = call i32 @caller( i32* %B )                ; <i32> [#uses=1]
        ret i32 %X
}

