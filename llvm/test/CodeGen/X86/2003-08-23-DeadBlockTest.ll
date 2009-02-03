; RUN: llvm-as < %s | llc -march=x86

define i32 @test() {
entry:
        ret i32 7
Test:           ; No predecessors!
        %A = call i32 @test( )          ; <i32> [#uses=1]
        %B = call i32 @test( )          ; <i32> [#uses=1]
        %C = add i32 %A, %B             ; <i32> [#uses=1]
        ret i32 %C
}

