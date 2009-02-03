; RUN: llvm-as < %s | llc -march=arm | grep {mov r0, r0, lsr #31}
; RUN: llvm-as < %s | llc -march=thumb | grep {lsr r0, r0, #31}

define i32 @test1(i32 %X) {
entry:
        icmp slt i32 %X, 0              ; <i1>:0 [#uses=1]
        zext i1 %0 to i32               ; <i32>:1 [#uses=1]
        ret i32 %1
}

