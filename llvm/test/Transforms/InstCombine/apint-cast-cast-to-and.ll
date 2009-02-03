; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep i41

define i61 @test1(i61 %X) {
        %Y = trunc i61 %X to i41 ;; Turn i61o an AND
        %Z = zext i41 %Y to i61
        ret i61 %Z
}

