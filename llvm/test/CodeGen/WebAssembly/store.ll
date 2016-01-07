; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic stores are assembled properly.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: sti32:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK-NEXT: i32.store $discard=, 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti32(i32 *%p, i32 %v) {
  store i32 %v, i32* %p
  ret void
}

; CHECK-LABEL: sti64:
; CHECK-NEXT: .param i32, i64{{$}}
; CHECK-NEXT: i64.store $discard=, 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @sti64(i64 *%p, i64 %v) {
  store i64 %v, i64* %p
  ret void
}

; CHECK-LABEL: stf32:
; CHECK-NEXT: .param i32, f32{{$}}
; CHECK-NEXT: f32.store $discard=, 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @stf32(float *%p, float %v) {
  store float %v, float* %p
  ret void
}

; CHECK-LABEL: stf64:
; CHECK-NEXT: .param i32, f64{{$}}
; CHECK-NEXT: f64.store $discard=, 0($0), $1{{$}}
; CHECK-NEXT: return{{$}}
define void @stf64(double *%p, double %v) {
  store double %v, double* %p
  ret void
}
