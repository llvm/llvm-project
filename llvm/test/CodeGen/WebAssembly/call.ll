; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic call operations assemble as expected.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare i32 @i32_nullary()
declare i32 @i32_unary(i32)
declare i32 @i32_binary(i32, i32)
declare i64 @i64_nullary()
declare float @float_nullary()
declare double @double_nullary()
declare void @void_nullary()

; CHECK-LABEL: call_i32_nullary:
; CHECK-NEXT: .result i32
; CHECK-NEXT: .local i32
; CHECK-NEXT: call $i32_nullary, push{{$}}
; CHECK-NEXT: set_local 0, pop
; CHECK-NEXT: return (get_local 0)
define i32 @call_i32_nullary() {
  %r = call i32 @i32_nullary()
  ret i32 %r
}

; CHECK-LABEL: call_i64_nullary:
; CHECK-NEXT: .result i64
; CHECK-NEXT: .local i64
; CHECK-NEXT: call $i64_nullary, push{{$}}
; CHECK-NEXT: set_local 0, pop
; CHECK-NEXT: return (get_local 0)
define i64 @call_i64_nullary() {
  %r = call i64 @i64_nullary()
  ret i64 %r
}

; CHECK-LABEL: call_float_nullary:
; CHECK-NEXT: .result f32
; CHECK-NEXT: .local f32
; CHECK-NEXT: call $float_nullary, push{{$}}
; CHECK-NEXT: set_local 0, pop
; CHECK-NEXT: return (get_local 0)
define float @call_float_nullary() {
  %r = call float @float_nullary()
  ret float %r
}

; CHECK-LABEL: call_double_nullary:
; CHECK-NEXT: .result f64
; CHECK-NEXT: .local f64
; CHECK-NEXT: call $double_nullary, push{{$}}
; CHECK-NEXT: set_local 0, pop
; CHECK-NEXT: return (get_local 0)
define double @call_double_nullary() {
  %r = call double @double_nullary()
  ret double %r
}

; CHECK-LABEL: call_void_nullary:
; CHECK-NEXT: call $void_nullary{{$}}
; CHECK-NEXT: return
define void @call_void_nullary() {
  call void @void_nullary()
  ret void
}

; CHECK-LABEL: call_i32_unary:
; CHECK-NEXT: .param i32
; CHECK-NEXT: .result i32
; CHECK-NEXT: .local i32, i32
; CHECK-NEXT: get_local push, 0
; CHECK-NEXT: set_local 1, pop
; CHECK-NEXT: call $i32_unary, push, (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop
; CHECK-NEXT: return (get_local 2)
define i32 @call_i32_unary(i32 %a) {
  %r = call i32 @i32_unary(i32 %a)
  ret i32 %r
}

; CHECK-LABEL: call_i32_binary:
; CHECK-NEXT: .param i32
; CHECK-NEXT: .param i32
; CHECK-NEXT: .result i32
; CHECK-NEXT: .local i32, i32, i32
; CHECK-NEXT: get_local push, 1
; CHECK-NEXT: set_local 2, pop
; CHECK-NEXT: get_local push, 0
; CHECK-NEXT: set_local 3, pop
; CHECK-NEXT: call $i32_binary, push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop
; CHECK-NEXT: return (get_local 4)
define i32 @call_i32_binary(i32 %a, i32 %b) {
  %r = call i32 @i32_binary(i32 %a, i32 %b)
  ret i32 %r
}

; CHECK-LABEL: call_indirect_void:
; CHECK-NEXT: .param i32
; CHECK-NEXT: .local i32
; CHECK-NEXT: get_local push, 0
; CHECK-NEXT: set_local 1, pop
; CHECK-NEXT: call_indirect (get_local 1){{$}}
; CHECK-NEXT: return
define void @call_indirect_void(void ()* %callee) {
  call void %callee()
  ret void
}

; CHECK-LABEL: call_indirect_i32:
; CHECK-NEXT: .param i32
; CHECK-NEXT: .result i32
; CHECK-NEXT: .local i32, i32
; CHECK-NEXT: get_local push, 0
; CHECK-NEXT: set_local 1, pop
; CHECK-NEXT: call_indirect (get_local 1), push{{$}}
; CHECK-NEXT: set_local 2, pop
; CHECK-NEXT: return (get_local 2)
define i32 @call_indirect_i32(i32 ()* %callee) {
  %t = call i32 %callee()
  ret i32 %t
}

; CHECK-LABEL: tail_call_void_nullary:
; CHECK-NEXT: call $void_nullary{{$}}
; CHECK-NEXT: return{{$}}
define void @tail_call_void_nullary() {
  tail call void @void_nullary()
  ret void
}

; CHECK-LABEL: fastcc_tail_call_void_nullary:
; CHECK-NEXT: call $void_nullary{{$}}
; CHECK-NEXT: return{{$}}
define void @fastcc_tail_call_void_nullary() {
  tail call fastcc void @void_nullary()
  ret void
}

; CHECK-LABEL: coldcc_tail_call_void_nullary:
; CHECK-NEXT: call $void_nullary
; CHECK-NEXT: return{{$}}
define void @coldcc_tail_call_void_nullary() {
  tail call coldcc void @void_nullary()
  ret void
}

; FIXME test the following:
;  - More argument combinations.
;  - Tail call.
;  - Interesting returns (struct, multiple).
;  - Vararg.
