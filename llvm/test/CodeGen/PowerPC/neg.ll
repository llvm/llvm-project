; RUN: llvm-as < %s | llc -march=ppc32 | grep neg

define i32 @test(i32 %X) {
        %Y = sub i32 0, %X              ; <i32> [#uses=1]
        ret i32 %Y
}

