; RUN: llvm-as < %s | opt -basicaa -gvn -instcombine |\
; RUN:   llvm-dis | grep {ret i32 0}

declare void @foo(i32*)

declare void @bar()

define i32 @test() {
        %A = alloca i32         ; <i32*> [#uses=3]
        call void @foo( i32* %A )
        %X = load i32* %A               ; <i32> [#uses=1]
        tail call void @bar( )
        %Y = load i32* %A               ; <i32> [#uses=1]
        %Z = sub i32 %X, %Y             ; <i32> [#uses=1]
        ret i32 %Z
}
