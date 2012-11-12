; RUN: llc < %s -march=ppc32 | not grep mul

define i32 @test1(i32 %a) {
        %tmp.1 = mul i32 %a, -2         ; <i32> [#uses=1]
        %tmp.2 = add i32 %tmp.1, 63             ; <i32> [#uses=1]
        ret i32 %tmp.2
}

