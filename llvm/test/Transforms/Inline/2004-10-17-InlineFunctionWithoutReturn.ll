; RUN: llvm-as < %s | opt -inline -disable-output

define i32 @test() {
        unwind
}

define i32 @caller() {
        %X = call i32 @test( )          ; <i32> [#uses=1]
        ret i32 %X
}

