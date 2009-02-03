; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:   grep {and i32 %Y, 8}

define i32 @test1(i8 %X) {
        %Y = zext i8 %X to i32          ; <i32> [#uses=1]
        %Z = and i32 %Y, 65544          ; <i32> [#uses=1]
        ret i32 %Z
}


