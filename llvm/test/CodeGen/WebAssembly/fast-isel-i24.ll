; RUN: llc < %s -O0 -wasm-keep-registers
; PR36564
; PR37546

; Test that fast-isel properly copes with i24 arguments and return types.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: add:
; CHECK-NEXT: .functype add (i32, i32) -> (i32){{$}}
; CHECK-NEXT: get_local	$push2=, 0{{$}}
; CHECK-NEXT: get_local	$push1=, 1{{$}}
; CHECK-NEXT: i32.add 	$push0=, $pop2, $pop1{{$}}
; CHECK-NEXT: end_function
define i24 @add(i24 %x, i24 %y) {
    %z = add i24 %x, %y
    ret i24 %z
}

; CHECK-LABEL: return_zero:
; CHECK-NEXT: .functype return_zero () -> (i32){{$}}
; CHECK-NEXT: i32.const	$push0=, 0{{$}}
; CHECK-NEXT: end_function
define i24 @return_zero() {
    ret i24 0
}
