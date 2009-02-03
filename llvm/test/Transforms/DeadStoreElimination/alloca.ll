; RUN: llvm-as < %s | opt -dse | llvm-dis | not grep DEAD

define void @test(i32* %Q) {
        %P = alloca i32         ; <i32*> [#uses=1]
        %DEAD = load i32* %Q            ; <i32> [#uses=1]
        store i32 %DEAD, i32* %P
        ret void
}

