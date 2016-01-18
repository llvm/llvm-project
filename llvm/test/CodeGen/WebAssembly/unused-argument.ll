; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Make sure that argument offsets are correct even if some arguments are unused.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: unused_first:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: .local i32{{$}}
; CHECK-NEXT: get_local 1{{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: return (get_local 2){{$}}
define i32 @unused_first(i32 %x, i32 %y) {
  ret i32 %y
}

; CHECK-LABEL: unused_second:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: .local i32{{$}}
; CHECK-NEXT: get_local 0{{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: return (get_local 2){{$}}
define i32 @unused_second(i32 %x, i32 %y) {
  ret i32 %x
}
