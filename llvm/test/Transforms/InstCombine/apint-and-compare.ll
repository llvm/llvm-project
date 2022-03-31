; RUN: opt < %s -passes=instcombine -S | grep and | count 2

; Should be optimized to one and.
define i1 @test1(i33 %a, i33 %b) {
        %tmp1 = and i33 %a, 65280
        %tmp3 = and i33 %b, 65280
        %tmp = icmp ne i33 %tmp1, %tmp3
        ret i1 %tmp
}

define i1 @test2(i999 %a, i999 %b) {
        %tmp1 = and i999 %a, 65280
        %tmp3 = and i999 %b, 65280
        %tmp = icmp ne i999 %tmp1, %tmp3
        ret i1 %tmp
}
