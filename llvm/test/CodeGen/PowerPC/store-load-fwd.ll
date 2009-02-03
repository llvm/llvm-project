; RUN: llvm-as < %s | llc -march=ppc32 | not grep lwz

define i32 @test(i32* %P) {
        store i32 1, i32* %P
        %V = load i32* %P               ; <i32> [#uses=1]
        ret i32 %V
}

