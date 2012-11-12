; This testcase causes instcombine to hang.
;
; RUN: opt < %s -instcombine

define void @test(i32 %X) {
        %reg117 = add i32 %X, 0         ; <i32> [#uses=0]
        ret void
}

