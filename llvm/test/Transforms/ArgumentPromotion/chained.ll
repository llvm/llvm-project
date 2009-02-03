; RUN: llvm-as < %s | opt -argpromotion -instcombine | llvm-dis | not grep load

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

