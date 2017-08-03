; RUN: llc -verify-machineinstrs < %s -march=ppc32 | not grep "ori\|lis"

; andi. r3, r3, 32769	
define i32 @test(i32 %X) {
        %Y = and i32 %X, 32769          ; <i32> [#uses=1]
        ret i32 %Y
}

; andis. r3, r3, 32769
define i32 @test2(i32 %X) {
        %Y = and i32 %X, -2147418112            ; <i32> [#uses=1]
        ret i32 %Y
}

