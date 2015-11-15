; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Make sure that argument offsets are correct even if some arguments are unused.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: unused_first:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: .local i32{{$}}
; CHECK-NEXT: return (get_local 1){{$}}
define i32 @unused_first(i32 %x, i32 %y) {
  ret i32 %y
}

; CHECK-LABEL: unused_second:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: .local i32{{$}}
; CHECK-NEXT: return (get_local 0){{$}}
define i32 @unused_second(i32 %x, i32 %y) {
  ret i32 %x
}

; CHECK-LABEL: call_something:
; CHECK-NEXT: call return_something, $discard{{$}}
; CHECK-NEXT: return{{$}}
declare i32 @return_something()
define void @call_something() {
  call i32 @return_something()
  ret void
}
