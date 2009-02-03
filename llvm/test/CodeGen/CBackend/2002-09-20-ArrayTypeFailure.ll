; RUN: llvm-as < %s | llc -march=c

define void @test() {
        %X = alloca [4 x i32]           ; <[4 x i32]*> [#uses=0]
        ret void
}

