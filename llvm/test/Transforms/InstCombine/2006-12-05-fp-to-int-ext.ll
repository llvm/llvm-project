; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep zext

; Never merge these two conversions, even though it's possible: this is
; significantly more expensive than the two conversions on some targets
; and it causes libgcc to be compile __fixunsdfdi into a recursive 
; function.
define i64 @test(double %D) {
        %A = fptoui double %D to i32            ; <i32> [#uses=1]
        %B = zext i32 %A to i64         ; <i64> [#uses=1]
        ret i64 %B
}

