; RUN: llc < %s -march=x86 | grep lea

declare i32 @foo()

define i32 @test() {
        %tmp.0 = tail call i32 @foo( )          ; <i32> [#uses=1]
        %tmp.1 = mul i32 %tmp.0, 9              ; <i32> [#uses=1]
        ret i32 %tmp.1
}

