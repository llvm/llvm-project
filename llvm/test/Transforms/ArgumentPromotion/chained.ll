; RUN: opt < %s -argpromotion -instcombine -S | not grep load
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

@G1 = constant i32 0            ; <i32*> [#uses=1]
@G2 = constant i32* @G1         ; <i32**> [#uses=1]

define internal i32 @test(i32** %X) {
        %Y = load i32** %X              ; <i32*> [#uses=1]
        %X.upgrd.1 = load i32* %Y               ; <i32> [#uses=1]
        ret i32 %X.upgrd.1
}

define i32 @caller(i32** %P) {
        %X = call i32 @test( i32** @G2 )                ; <i32> [#uses=1]
        ret i32 %X
}

