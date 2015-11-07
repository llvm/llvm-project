; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic loads are assembled properly.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: ldi32:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: .local i32, i32{{$}}
; CHECK-NEXT: get_local push, 0{{$}}
; CHECK-NEXT: set_local 1, pop{{$}}
; CHECK-NEXT: i32.load push, (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: return (get_local 2){{$}}
define i32 @ldi32(i32 *%p) {
  %v = load i32, i32* %p
  ret i32 %v
}

; CHECK-LABEL: ldi64:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: .local i32, i64{{$}}
; CHECK-NEXT: get_local push, 0{{$}}
; CHECK-NEXT: set_local 1, pop{{$}}
; CHECK-NEXT: i64.load push, (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: return (get_local 2){{$}}
define i64 @ldi64(i64 *%p) {
  %v = load i64, i64* %p
  ret i64 %v
}

; CHECK-LABEL: ldf32:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result f32{{$}}
; CHECK-NEXT: .local i32, f32{{$}}
; CHECK-NEXT: get_local push, 0{{$}}
; CHECK-NEXT: set_local 1, pop{{$}}
; CHECK-NEXT: f32.load push, (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: return (get_local 2){{$}}
define float @ldf32(float *%p) {
  %v = load float, float* %p
  ret float %v
}

; CHECK-LABEL: ldf64:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result f64{{$}}
; CHECK-NEXT: .local i32, f64{{$}}
; CHECK-NEXT: get_local push, 0{{$}}
; CHECK-NEXT: set_local 1, pop{{$}}
; CHECK-NEXT: f64.load push, (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: return (get_local 2){{$}}
define double @ldf64(double *%p) {
  %v = load double, double* %p
  ret double %v
}
