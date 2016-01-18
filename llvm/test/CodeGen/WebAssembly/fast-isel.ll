; RUN: llc < %s -asm-verbose=false \
; RUN:   -fast-isel -fast-isel-abort=1 -verify-machineinstrs \
; RUN:   | FileCheck %s

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; This tests very minimal fast-isel functionality.

; CHECK-LABEL: immediate_f32:
; CHECK: f32.const 0x1.4p1{{$}}
define float @immediate_f32() {
  ret float 2.5
}

; CHECK-LABEL: immediate_f64:
; CHECK: f64.const 0x1.4p1{{$}}
define double @immediate_f64() {
  ret double 2.5
}
