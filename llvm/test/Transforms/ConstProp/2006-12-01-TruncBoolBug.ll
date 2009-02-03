; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:   grep {ret i1 false}
define i1 @test() {
        %X = trunc i32 320 to i1                ; <i1> [#uses=1]
        ret i1 %X
}

