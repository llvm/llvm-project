; RUN: opt < %s -instcombine -S | \
; RUN:    grep and | count 1

; Should be optimized to one and.
define i1 @test1(i32 %a, i32 %b) {
        %tmp1 = and i32 %a, 65280               ; <i32> [#uses=1]
        %tmp3 = and i32 %b, 65280               ; <i32> [#uses=1]
        %tmp = icmp ne i32 %tmp1, %tmp3         ; <i1> [#uses=1]
        ret i1 %tmp
}

