; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep -- -65536

define i1 @test(i32 %tmp.124) {
        %tmp.125 = shl i32 %tmp.124, 8          ; <i32> [#uses=1]
        %tmp.126.mask = and i32 %tmp.125, -16777216             ; <i32> [#uses=1]
        %tmp.128 = icmp eq i32 %tmp.126.mask, 167772160         ; <i1> [#uses=1]
        ret i1 %tmp.128
}

