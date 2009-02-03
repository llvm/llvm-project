; The register allocator can commute two-address instructions to avoid
; insertion of register-register copies.

; Make sure there are only 3 mov's for each testcase
; RUN: llvm-as < %s | llc -march=x86 -x86-asm-syntax=intel | \
; RUN:   grep {\\\<mov\\\>} | count 6


target triple = "i686-pc-linux-gnu"
@G = external global i32                ; <i32*> [#uses=2]

declare void @ext(i32)

define i32 @add_test(i32 %X, i32 %Y) {
        %Z = add i32 %X, %Y             ; <i32> [#uses=1]
        store i32 %Z, i32* @G
        ret i32 %X
}

define i32 @xor_test(i32 %X, i32 %Y) {
        %Z = xor i32 %X, %Y             ; <i32> [#uses=1]
        store i32 %Z, i32* @G
        ret i32 %X
}

