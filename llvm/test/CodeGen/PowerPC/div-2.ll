; RUN: llc < %s -march=ppc32 | not grep srawi 
; RUN: llc < %s -march=ppc32 | grep blr

define i32 @test1(i32 %X) {
        %Y = and i32 %X, 15             ; <i32> [#uses=1]
        %Z = sdiv i32 %Y, 4             ; <i32> [#uses=1]
        ret i32 %Z
}

define i32 @test2(i32 %W) {
        %X = and i32 %W, 15             ; <i32> [#uses=1]
        %Y = sub i32 16, %X             ; <i32> [#uses=1]
        %Z = sdiv i32 %Y, 4             ; <i32> [#uses=1]
        ret i32 %Z
}

define i32 @test3(i32 %W) {
        %X = and i32 %W, 15             ; <i32> [#uses=1]
        %Y = sub i32 15, %X             ; <i32> [#uses=1]
        %Z = sdiv i32 %Y, 4             ; <i32> [#uses=1]
        ret i32 %Z
}

define i32 @test4(i32 %W) {
        %X = and i32 %W, 2              ; <i32> [#uses=1]
        %Y = sub i32 5, %X              ; <i32> [#uses=1]
        %Z = sdiv i32 %Y, 2             ; <i32> [#uses=1]
        ret i32 %Z
}

