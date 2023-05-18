; RUN: opt < %s -passes=instcombine -S | \
; RUN:   not grep "ret i1 false"

define i1 @test(i64 %tmp.169) {
        %tmp.1710 = lshr i64 %tmp.169, 1                ; <i64> [#uses=1]
        %tmp.1912 = icmp ugt i64 %tmp.1710, 0           ; <i1> [#uses=1]
        ret i1 %tmp.1912
}

