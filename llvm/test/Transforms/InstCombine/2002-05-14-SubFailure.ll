; Instcombine was missing a test that caused it to make illegal transformations
; sometimes.  In this case, it transforms the sub into an add:
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep sub
;
define i32 @test(i32 %i, i32 %j) {
        %A = mul i32 %i, %j
        %B = sub i32 2, %A
        ret i32 %B
}

