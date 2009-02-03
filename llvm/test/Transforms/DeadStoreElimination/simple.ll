; RUN: llvm-as < %s | opt -dse | llvm-dis | not grep DEAD

define void @test(i32* %Q, i32* %P) {
        %DEAD = load i32* %Q            ; <i32> [#uses=1]
        store i32 %DEAD, i32* %P
        store i32 0, i32* %P
        ret void
}

