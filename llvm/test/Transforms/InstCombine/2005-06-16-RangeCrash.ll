; RUN: llvm-as < %s | opt -instcombine -disable-output
; PR585

define i1 @test() {
        %tmp.26 = sdiv i32 0, -2147483648               ; <i32> [#uses=1]
        %tmp.27 = icmp eq i32 %tmp.26, 0                ; <i1> [#uses=1]
        ret i1 %tmp.27
}

