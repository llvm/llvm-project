; RUN: llc < %s -march=arm -mattr=+v6 | grep uxtb | count 1
; RUN: llc < %s -march=arm -mattr=+v6 | grep uxtab | count 1
; RUN: llc < %s -march=arm -mattr=+v6 | grep uxth | count 1

define zeroext i8 @test1(i32 %A.u) {
    %B.u = trunc i32 %A.u to i8
    ret i8 %B.u
}

define zeroext i32 @test2(i32 %A.u, i32 %B.u) {
    %C.u = trunc i32 %B.u to i8
    %D.u = zext i8 %C.u to i32
    %E.u = add i32 %A.u, %D.u
    ret i32 %E.u
}

define zeroext i32 @test3(i32 %A.u) {
    %B.u = lshr i32 %A.u, 8
    %C.u = shl i32 %A.u, 24
    %D.u = or i32 %B.u, %C.u
    %E.u = trunc i32 %D.u to i16
    %F.u = zext i16 %E.u to i32
    ret i32 %F.u
}
