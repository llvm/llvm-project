; REQUIRES: asserts
; The old instruction selector used to load all arguments to a call up in 
; registers, then start pushing them all onto the stack.  This is bad news as
; it makes a ton of annoying overlapping live ranges.  This code should not
; cause spills!
;
; RUN: llc < %s -march=x86 -stats 2>&1 | FileCheck %s

; CHECK-NOT: spilled

target datalayout = "e-p:32:32"

define i32 @test(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) {
        ret i32 0
}

define i32 @main() {
        %X = call i32 @test( i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10 )            ; <i32> [#uses=1]
        ret i32 %X
}

