; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep false
;
; This actually looks like a constant propagation bug

%X = type { [10 x i32], float }

define i1 @test() {
        %A = getelementptr %X* null, i64 0, i32 0, i64 0                ; <i32*> [#uses=1]
        %B = icmp ne i32* %A, null              ; <i1> [#uses=1]
        ret i1 %B
}

