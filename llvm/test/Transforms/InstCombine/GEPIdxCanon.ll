; RUN: llvm-as < %s | opt -instcombine -gvn -instcombine | \
; RUN:    llvm-dis | not grep getelementptr

define i1 @test(i32* %A) {
        %B = getelementptr i32* %A, i32 1               ; <i32*> [#uses=1]
        %C = getelementptr i32* %A, i64 1               ; <i32*> [#uses=1]
        %V = icmp eq i32* %B, %C                ; <i1> [#uses=1]
        ret i1 %V
}

