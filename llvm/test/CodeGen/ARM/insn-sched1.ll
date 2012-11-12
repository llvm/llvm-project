; RUN: llc < %s -march=arm -mattr=+v6
; RUN: llc < %s -mtriple=arm-apple-ios -mattr=+v6 |\
; RUN:   grep mov | count 3

define i32 @test(i32 %x) {
        %tmp = trunc i32 %x to i16              ; <i16> [#uses=1]
        %tmp2 = call i32 @f( i32 1, i16 %tmp )             ; <i32> [#uses=1]
        ret i32 %tmp2
}

declare i32 @f(i32, i16)
