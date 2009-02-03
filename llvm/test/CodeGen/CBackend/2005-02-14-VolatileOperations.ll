; RUN: llvm-as < %s | llc -march=c | grep volatile

define void @test(i32* %P) {
        %X = volatile load i32* %P              ; <i32> [#uses=1]
        volatile store i32 %X, i32* %P
        ret void
}

