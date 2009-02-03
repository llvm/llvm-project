; test that casted mallocs get converted to malloc of the right type
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:    not grep bitcast

; The target datalayout is important for this test case. We have to tell 
; instcombine that the ABI alignment for a long is 4-bytes, not 8, otherwise
; it won't do the transform.
target datalayout = "e-i64:32:64"

define i32* @test(i32 %size) {
        %X = malloc i64, i32 %size              ; <i64*> [#uses=1]
        %ret = bitcast i64* %X to i32*          ; <i32*> [#uses=1]
        ret i32* %ret
}

