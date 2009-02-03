; Ensure constant propogation of 'not' instructions is working correctly.

; RUN: llvm-as < %s | opt -constprop -die | llvm-dis | not grep xor

define i32 @test1() {
        %R = xor i32 4, -1              ; <i32> [#uses=1]
        ret i32 %R
}

define i32 @test2() {
        %R = xor i32 -23, -1            ; <i32> [#uses=1]
        ret i32 %R
}

define i1 @test3() {
        %R = xor i1 true, true          ; <i1> [#uses=1]
        ret i1 %R
}

