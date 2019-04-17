; This bug has to do with the fact that constant propagation was implemented in
; terms of _logical_ not (! in C) instead of _bitwise_ not (~ in C).  This was
; due to a spec change.

; Fix #2: The unary not instruction now no longer exists. Change to xor.

; RUN: opt < %s -constprop -S | \
; RUN:   not grep "i32 0"

define i32 @test1() {
        %R = xor i32 123, -1            ; <i32> [#uses=1]
        ret i32 %R
}

define i32 @test2() {
        %R = xor i32 -123, -1           ; <i32> [#uses=1]
        ret i32 %R
}

