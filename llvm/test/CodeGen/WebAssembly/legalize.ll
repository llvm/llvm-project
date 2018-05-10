; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -disable-wasm-explicit-locals | FileCheck %s

; Test various types and operators that need to be legalized.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: shl_i3:
; CHECK: i32.const   $push0=, 7{{$}}
; CHECK: i32.and     $push1=, $1, $pop0{{$}}
; CHECK: i32.shl     $push2=, $0, $pop1{{$}}
define i3 @shl_i3(i3 %a, i3 %b, i3* %p) {
  %t = shl i3 %a, %b
  ret i3 %t
}

; CHECK-LABEL: shl_i53:
; CHECK: i64.const   $push0=, 9007199254740991{{$}}
; CHECK: i64.and     $push1=, $1, $pop0{{$}}
; CHECK: i64.shl     $push2=, $0, $pop1{{$}}
define i53 @shl_i53(i53 %a, i53 %b, i53* %p) {
  %t = shl i53 %a, %b
  ret i53 %t
}

; CHECK-LABEL: sext_in_reg_i32_i64:
; CHECK: i64.shl
; CHECK: i64.shr_s
define i64 @sext_in_reg_i32_i64(i64 %a) {
  %b = shl i64 %a, 32
  %c = ashr i64 %b, 32
  ret i64 %c
}

; CHECK-LABEL: fpext_f32_f64:
; CHECK: f32.load $push0=, 0($0){{$}}
; CHECK: f64.promote/f32 $push1=, $pop0{{$}}
; CHECK: return $pop1{{$}}
define double @fpext_f32_f64(float *%p) {
  %v = load float, float* %p
  %e = fpext float %v to double
  ret double %e
}

; CHECK-LABEL: fpconv_f64_f32:
; CHECK: f64.load $push0=, 0($0){{$}}
; CHECK: f32.demote/f64 $push1=, $pop0{{$}}
; CHECK: return $pop1{{$}}
define float @fpconv_f64_f32(double *%p) {
  %v = load double, double* %p
  %e = fptrunc double %v to float
  ret float %e
}

; Check that big shifts work. This generates a big pile of code from the
; legalizer; the main thing here is that we don't abort.

; CHECK-LABEL: bigshift:
define i1024 @bigshift(i1024 %a, i1024 %b) {
    %c = shl i1024 %a, %b
    ret i1024 %c
}
