; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep shl

define i1 @test(i32 %X, i8 %A) {
        %shift.upgrd.1 = zext i8 %A to i32              ; <i32> [#uses=1]
        %B = lshr i32 %X, %shift.upgrd.1                ; <i32> [#uses=1]
        %D = trunc i32 %B to i1         ; <i1> [#uses=1]
        ret i1 %D
}

