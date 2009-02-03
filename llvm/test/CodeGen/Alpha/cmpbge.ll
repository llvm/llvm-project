; RUN: llvm-as < %s | llc -march=alpha | grep cmpbge | count 2

define i1 @test1(i64 %A, i64 %B) {
        %C = and i64 %A, 255            ; <i64> [#uses=1]
        %D = and i64 %B, 255            ; <i64> [#uses=1]
        %E = icmp uge i64 %C, %D                ; <i1> [#uses=1]
        ret i1 %E
}

define i1 @test2(i64 %a, i64 %B) {
        %A = shl i64 %a, 1              ; <i64> [#uses=1]
        %C = and i64 %A, 254            ; <i64> [#uses=1]
        %D = and i64 %B, 255            ; <i64> [#uses=1]
        %E = icmp uge i64 %C, %D                ; <i1> [#uses=1]
        ret i1 %E
}
