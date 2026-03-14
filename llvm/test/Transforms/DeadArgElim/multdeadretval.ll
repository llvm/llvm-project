; This test sees if return values (and arguments) are properly removed when they
; are unused. All unused values are typed i16, so we can easily check. We also
; run instcombine to fold insert/extractvalue chains and we run dce to clean up
; any remaining dead stuff.
; RUN: opt < %s -passes='deadargelim,function(instcombine),function(dce)' -S | not grep i16

define internal {i16, i32} @test(i16 %DEADARG) {
        %A = insertvalue {i16,i32} undef, i16 1, 0
        %B = insertvalue {i16,i32} %A, i32 1001, 1
        ret {i16,i32} %B
}

define internal {i32, i16} @test2() {
        %DEAD = call i16 @test4()
        %A = insertvalue {i32,i16} undef, i32 1, 0
        %B = insertvalue {i32,i16} %A, i16 %DEAD, 1
        ret {i32,i16} %B
}

; Dead argument, used to check if the second result of test2 is dead even when
; it's used as a dead argument
define internal i32 @test3(i16 %A) {
        %ret = call {i16, i32} @test( i16 %A )                ; <i32> [#uses=0]
        %DEAD = extractvalue {i16, i32} %ret, 0
        %LIVE = extractvalue {i16, i32} %ret, 1
        ret i32 %LIVE
}

define internal i16 @test4() {
        ret i16 0
}

; Multiple return values, multiple live return values
define internal {i32, i32, i16} @test5() {
        %A = insertvalue {i32,i32,i16} undef, i32 1, 0
        %B = insertvalue {i32,i32,i16} %A, i32 2, 1
        %C = insertvalue {i32,i32,i16} %B, i16 3, 2
        ret {i32, i32, i16} %C
}

; Nested return values
define internal {{i32}, {i16, i16}} @test6() {
        %A = insertvalue {{i32}, {i16, i16}} undef, i32 1, 0, 0
        %B = insertvalue {{i32}, {i16, i16}} %A, i16 2, 1, 0
        %C = insertvalue {{i32}, {i16, i16}} %B, i16 3, 1, 1
        ret {{i32}, {i16, i16}} %C
}

define i32 @main() {
        %ret = call {i32, i16} @test2()                ; <i32> [#uses=1]
        %LIVE = extractvalue {i32, i16} %ret, 0
        %DEAD = extractvalue {i32, i16} %ret, 1
        %Y = add i32 %LIVE, -123           ; <i32> [#uses=1]
        %LIVE2 = call i32 @test3(i16 %DEAD)                ; <i32> [#uses=1]
        %Z = add i32 %LIVE2, %Y           ; <i32> [#uses=1]
        %ret1 = call { i32, i32, i16 } @test5 ()
        %LIVE3 = extractvalue { i32, i32, i16} %ret1, 0
        %LIVE4 = extractvalue { i32, i32, i16} %ret1, 1
        %DEAD2 = extractvalue { i32, i32, i16} %ret1, 2
        %V = add i32 %LIVE3, %LIVE4
        %W = add i32 %Z, %V
        %ret2 = call { { i32 }, { i16, i16 } } @test6 ()
        %LIVE5 = extractvalue { { i32 }, { i16, i16 } } %ret2, 0, 0
        %DEAD3 = extractvalue { { i32 }, { i16, i16 } } %ret2, 1, 0
        %DEAD4 = extractvalue { { i32 }, { i16, i16 } } %ret2, 1, 1
        %Q = add i32 %W, %LIVE5
        ret i32 %Q
}
