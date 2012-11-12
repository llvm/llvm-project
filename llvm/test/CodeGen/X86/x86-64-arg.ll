; RUN: llc < %s | grep "movl	%edi, %eax"
; The input value is already sign extended, don't re-extend it.
; This testcase corresponds to:
;   int test(short X) { return (int)X; }

target datalayout = "e-p:64:64"
target triple = "x86_64-apple-darwin8"


define i32 @test(i16 signext  %X) {
entry:
        %tmp12 = sext i16 %X to i32             ; <i32> [#uses=1]
        ret i32 %tmp12
}

