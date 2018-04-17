; RUN: llc < %s -O0
; PR36564

; Test that fast-isel properly copes with i24 arguments and return types.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

define i24 @add(i24 %x, i24 %y) {
    %z = add i24 %x, %y
    ret i24 %z
}

define i24 @return_zero() {
    ret i24 0
}
