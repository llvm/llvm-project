; Ensure constant propagation of remainder instructions is working correctly.

; RUN: llvm-as < %s | opt -constprop -die | llvm-dis | not grep rem

define i32 @test1() {
        %R = srem i32 4, 3              ; <i32> [#uses=1]
        ret i32 %R
}

define i32 @test2() {
        %R = srem i32 123, -23          ; <i32> [#uses=1]
        ret i32 %R
}

define float @test3() {
        %R = frem float 0x4028E66660000000, 0x405ECDA1C0000000          ; <float> [#uses=1]
        ret float %R
}

define double @test4() {
        %R = frem double 0x4073833BEE07AFF8, 0x4028AAABB2A0D19C         ; <double> [#uses=1]
        ret double %R
}

