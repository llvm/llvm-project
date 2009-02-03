; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep trunc

target datalayout = "e-p:32:32"

define i32* @test(i32* %X, i64 %Idx) {
        %R = getelementptr i32* %X, i64 %Idx            ; <i32*> [#uses=1]
        ret i32* %R
}

