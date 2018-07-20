; RUN: llc < %s -O0
; PR36564
; PR37546

; Test that fast-isel properly copes with i256 arguments and return types.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: add:
; CHECK-NEXT: .param      i32, i64, i64, i64, i64, i64, i64, i64, i64{{$}}
; CHECK-NOT:  .result
; CHECK: end_function
define i256 @add(i256 %x, i256 %y) {
    %z = add i256 %x, %y
    ret i256 %z
}

; CHECK-LABEL: return_zero:
; CHECK-NEXT: .param      i32{{$}}
; CHECK-NOT: .result
; CHECK: end_function
define i256 @return_zero() {
    ret i256 0
}

; CHECK-LABEL: return_zero_with_params:
; CHECK-NEXT: .param      i32, f32{{$}}
; CHECK-NOT: .result
; CHECK: end_function
define i256 @return_zero_with_params(float %x) {
    ret i256 0
}
