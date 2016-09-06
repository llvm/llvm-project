; RUN: llc < %s -asm-verbose=false \
; RUN:   -fast-isel -fast-isel-abort=1 -verify-machineinstrs \
; RUN:   | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; This tests very minimal fast-isel functionality.

; CHECK-LABEL: immediate_f32:
; CHECK: f32.const $push{{[0-9]+}}=, 0x1.4p1{{$}}
define float @immediate_f32() {
  ret float 2.5
}

; CHECK-LABEL: immediate_f64:
; CHECK: f64.const $push{{[0-9]+}}=, 0x1.4p1{{$}}
define double @immediate_f64() {
  ret double 2.5
}

; CHECK-LABEL: bitcast_i32_f32:
; CHECK: i32.reinterpret/f32 $push{{[0-9]+}}=, $0{{$}}
define i32 @bitcast_i32_f32(float %x) {
  %y = bitcast float %x to i32
  ret i32 %y
}

; CHECK-LABEL: bitcast_f32_i32:
; CHECK: f32.reinterpret/i32 $push{{[0-9]+}}=, $0{{$}}
define float @bitcast_f32_i32(i32 %x) {
  %y = bitcast i32 %x to float
  ret float %y
}

; CHECK-LABEL: bitcast_i64_f64:
; CHECK: i64.reinterpret/f64 $push{{[0-9]+}}=, $0{{$}}
define i64 @bitcast_i64_f64(double %x) {
  %y = bitcast double %x to i64
  ret i64 %y
}

; CHECK-LABEL: bitcast_f64_i64:
; CHECK: f64.reinterpret/i64 $push{{[0-9]+}}=, $0{{$}}
define double @bitcast_f64_i64(i64 %x) {
  %y = bitcast i64 %x to double
  ret double %y
}
