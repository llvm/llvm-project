; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic conversion operations assemble as expected.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: i32_wrap_i64:
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: .local i64, i32{{$}}
; CHECK-NEXT: get_local 0{{$}}
; CHECK-NEXT: set_local 1, pop{{$}}
; CHECK-NEXT: i32.wrap/i64 (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: return (get_local 2){{$}}
define i32 @i32_wrap_i64(i64 %x) {
  %a = trunc i64 %x to i32
  ret i32 %a
}

; CHECK-LABEL: i64_extend_s_i32:
; CHECK-NEXT: .param i32
; CHECK-NEXT: .result i64
; CHECK-NEXT: .local i32, i64{{$}}
; CHECK-NEXT: get_local 0{{$}}
; CHECK-NEXT: set_local 1, pop{{$}}
; CHECK-NEXT: i64.extend_s/i32 (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: return (get_local 2){{$}}
define i64 @i64_extend_s_i32(i32 %x) {
  %a = sext i32 %x to i64
  ret i64 %a
}

; CHECK-LABEL: i64_extend_u_i32:
; CHECK-NEXT: .param i32
; CHECK-NEXT: .result i64
; CHECK-NEXT: .local i32, i64{{$}}
; CHECK-NEXT: get_local 0{{$}}
; CHECK-NEXT: set_local 1, pop{{$}}
; CHECK-NEXT: i64.extend_u/i32 (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: return (get_local 2){{$}}
define i64 @i64_extend_u_i32(i32 %x) {
  %a = zext i32 %x to i64
  ret i64 %a
}

; CHECK-LABEL: i32_trunc_s_f32:
; CHECK-NEXT: .param f32
; CHECK-NEXT: .result i32
; CHECK-NEXT: .local f32, i32{{$}}
; CHECK-NEXT: get_local 0{{$}}
; CHECK-NEXT: set_local 1, pop{{$}}
; CHECK-NEXT: i32.trunc_s/f32 (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: return (get_local 2){{$}}
define i32 @i32_trunc_s_f32(float %x) {
  %a = fptosi float %x to i32
  ret i32 %a
}

; CHECK-LABEL: i32_trunc_u_f32:
; CHECK-NEXT: .param f32
; CHECK-NEXT: .result i32
; CHECK-NEXT: .local f32, i32{{$}}
; CHECK-NEXT: get_local 0{{$}}
; CHECK-NEXT: set_local 1, pop{{$}}
; CHECK-NEXT: i32.trunc_u/f32 (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: return (get_local 2){{$}}
define i32 @i32_trunc_u_f32(float %x) {
  %a = fptoui float %x to i32
  ret i32 %a
}

; CHECK-LABEL: i32_trunc_s_f64:
; CHECK-NEXT: .param f64
; CHECK-NEXT: .result i32
; CHECK-NEXT: .local f64, i32{{$}}
; CHECK-NEXT: get_local 0{{$}}
; CHECK-NEXT: set_local 1, pop{{$}}
; CHECK-NEXT: i32.trunc_s/f64 (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: return (get_local 2){{$}}
define i32 @i32_trunc_s_f64(double %x) {
  %a = fptosi double %x to i32
  ret i32 %a
}

; CHECK-LABEL: i32_trunc_u_f64:
; CHECK-NEXT: .param f64
; CHECK-NEXT: .result i32
; CHECK-NEXT: .local f64, i32{{$}}
; CHECK-NEXT: get_local 0{{$}}
; CHECK-NEXT: set_local 1, pop{{$}}
; CHECK-NEXT: i32.trunc_u/f64 (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: return (get_local 2){{$}}
define i32 @i32_trunc_u_f64(double %x) {
  %a = fptoui double %x to i32
  ret i32 %a
}

; CHECK-LABEL: i64_trunc_s_f32:
; CHECK-NEXT: .param f32
; CHECK-NEXT: .result i64
; CHECK-NEXT: .local f32, i64{{$}}
; CHECK-NEXT: get_local 0{{$}}
; CHECK-NEXT: set_local 1, pop{{$}}
; CHECK-NEXT: i64.trunc_s/f32 (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: return (get_local 2){{$}}
define i64 @i64_trunc_s_f32(float %x) {
  %a = fptosi float %x to i64
  ret i64 %a
}

; CHECK-LABEL: i64_trunc_u_f32:
; CHECK-NEXT: .param f32
; CHECK-NEXT: .result i64
; CHECK-NEXT: .local f32, i64{{$}}
; CHECK-NEXT: get_local 0{{$}}
; CHECK-NEXT: set_local 1, pop{{$}}
; CHECK-NEXT: i64.trunc_u/f32 (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: return (get_local 2){{$}}
define i64 @i64_trunc_u_f32(float %x) {
  %a = fptoui float %x to i64
  ret i64 %a
}

; CHECK-LABEL: i64_trunc_s_f64:
; CHECK-NEXT: .param f64
; CHECK-NEXT: .result i64
; CHECK-NEXT: .local f64, i64{{$}}
; CHECK-NEXT: get_local 0{{$}}
; CHECK-NEXT: set_local 1, pop{{$}}
; CHECK-NEXT: i64.trunc_s/f64 (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: return (get_local 2){{$}}
define i64 @i64_trunc_s_f64(double %x) {
  %a = fptosi double %x to i64
  ret i64 %a
}

; CHECK-LABEL: i64_trunc_u_f64:
; CHECK-NEXT: .param f64
; CHECK-NEXT: .result i64
; CHECK-NEXT: .local f64, i64{{$}}
; CHECK-NEXT: get_local 0{{$}}
; CHECK-NEXT: set_local 1, pop{{$}}
; CHECK-NEXT: i64.trunc_u/f64 (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: return (get_local 2){{$}}
define i64 @i64_trunc_u_f64(double %x) {
  %a = fptoui double %x to i64
  ret i64 %a
}

; CHECK-LABEL: f32_convert_s_i32:
; CHECK-NEXT: .param i32
; CHECK-NEXT: .result f32
; CHECK-NEXT: .local i32, f32{{$}}
; CHECK-NEXT: get_local 0{{$}}
; CHECK-NEXT: set_local 1, pop{{$}}
; CHECK-NEXT: f32.convert_s/i32 (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: return (get_local 2){{$}}
define float @f32_convert_s_i32(i32 %x) {
  %a = sitofp i32 %x to float
  ret float %a
}

; CHECK-LABEL: f32_convert_u_i32:
; CHECK-NEXT: .param i32
; CHECK-NEXT: .result f32
; CHECK-NEXT: .local i32, f32{{$}}
; CHECK-NEXT: get_local 0{{$}}
; CHECK-NEXT: set_local 1, pop{{$}}
; CHECK-NEXT: f32.convert_u/i32 (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: return (get_local 2){{$}}
define float @f32_convert_u_i32(i32 %x) {
  %a = uitofp i32 %x to float
  ret float %a
}

; CHECK-LABEL: f64_convert_s_i32:
; CHECK-NEXT: .param i32
; CHECK-NEXT: .result f64
; CHECK-NEXT: .local i32, f64{{$}}
; CHECK-NEXT: get_local 0{{$}}
; CHECK-NEXT: set_local 1, pop{{$}}
; CHECK-NEXT: f64.convert_s/i32 (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: return (get_local 2){{$}}
define double @f64_convert_s_i32(i32 %x) {
  %a = sitofp i32 %x to double
  ret double %a
}

; CHECK-LABEL: f64_convert_u_i32:
; CHECK-NEXT: .param i32
; CHECK-NEXT: .result f64
; CHECK-NEXT: .local i32, f64{{$}}
; CHECK-NEXT: get_local 0{{$}}
; CHECK-NEXT: set_local 1, pop{{$}}
; CHECK-NEXT: f64.convert_u/i32 (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: return (get_local 2){{$}}
define double @f64_convert_u_i32(i32 %x) {
  %a = uitofp i32 %x to double
  ret double %a
}

; CHECK-LABEL: f32_convert_s_i64:
; CHECK-NEXT: .param i64
; CHECK-NEXT: .result f32
; CHECK-NEXT: .local i64, f32{{$}}
; CHECK-NEXT: get_local 0{{$}}
; CHECK-NEXT: set_local 1, pop{{$}}
; CHECK-NEXT: f32.convert_s/i64 (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: return (get_local 2){{$}}
define float @f32_convert_s_i64(i64 %x) {
  %a = sitofp i64 %x to float
  ret float %a
}

; CHECK-LABEL: f32_convert_u_i64:
; CHECK-NEXT: .param i64
; CHECK-NEXT: .result f32
; CHECK-NEXT: .local i64, f32{{$}}
; CHECK-NEXT: get_local 0{{$}}
; CHECK-NEXT: set_local 1, pop{{$}}
; CHECK-NEXT: f32.convert_u/i64 (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: return (get_local 2){{$}}
define float @f32_convert_u_i64(i64 %x) {
  %a = uitofp i64 %x to float
  ret float %a
}

; CHECK-LABEL: f64_convert_s_i64:
; CHECK-NEXT: .param i64
; CHECK-NEXT: .result f64
; CHECK-NEXT: .local i64, f64{{$}}
; CHECK-NEXT: get_local 0{{$}}
; CHECK-NEXT: set_local 1, pop{{$}}
; CHECK-NEXT: f64.convert_s/i64 (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: return (get_local 2){{$}}
define double @f64_convert_s_i64(i64 %x) {
  %a = sitofp i64 %x to double
  ret double %a
}

; CHECK-LABEL: f64_convert_u_i64:
; CHECK-NEXT: .param i64
; CHECK-NEXT: .result f64
; CHECK-NEXT: .local i64, f64{{$}}
; CHECK-NEXT: get_local 0{{$}}
; CHECK-NEXT: set_local 1, pop{{$}}
; CHECK-NEXT: f64.convert_u/i64 (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: return (get_local 2){{$}}
define double @f64_convert_u_i64(i64 %x) {
  %a = uitofp i64 %x to double
  ret double %a
}

; CHECK-LABEL: f64_promote_f32:
; CHECK-NEXT: .param f32
; CHECK-NEXT: .result f64
; CHECK-NEXT: .local f32, f64{{$}}
; CHECK-NEXT: get_local 0{{$}}
; CHECK-NEXT: set_local 1, pop{{$}}
; CHECK-NEXT: f64.promote/f32 (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: return (get_local 2){{$}}
define double @f64_promote_f32(float %x) {
  %a = fpext float %x to double
  ret double %a
}

; CHECK-LABEL: f32_demote_f64:
; CHECK-NEXT: .param f64
; CHECK-NEXT: .result f32
; CHECK-NEXT: .local f64, f32{{$}}
; CHECK-NEXT: get_local 0{{$}}
; CHECK-NEXT: set_local 1, pop{{$}}
; CHECK-NEXT: f32.demote/f64 (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: return (get_local 2){{$}}
define float @f32_demote_f64(double %x) {
  %a = fptrunc double %x to float
  ret float %a
}
