; RUN: opt < %s -instcombine -disable-output
; PR577

define i1 @test() {
        %tmp.3 = shl i32 0, 41          ; <i32> [#uses=1]
        %tmp.4 = icmp ne i32 %tmp.3, 0          ; <i1> [#uses=1]
        ret i1 %tmp.4
}

