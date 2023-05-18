; RUN: llvm-as %s -o /dev/null
; RUN: verify-uselistorder %s


define i32 @test(i1 %C, i32 %V1, i32 %V2) {
        %X = select i1 true, i1 false, i1 true          ; <i1> [#uses=1]
        %V = select i1 %X, i32 %V1, i32 %V2             ; <i32> [#uses=1]
        ret i32 %V
}

