; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:   not grep i8 

define i32 @test1(i32 %X) {
        %Y = trunc i32 %X to i8         ; <i8> [#uses=1]
        %Z = zext i8 %Y to i32          ; <i32> [#uses=1]
        ret i32 %Z
}

