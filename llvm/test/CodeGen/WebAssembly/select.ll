; RUN: llc < %s -asm-verbose=false | FileCheck %s
; RUN: llc < %s -asm-verbose=false -fast-isel | FileCheck %s

; Test that wasm select instruction is selected from LLVM select instruction.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: select_i32_bool:
; CHECK-NEXT: .param     i32, i32, i32{{$}}
; CHECK-NEXT: .result    i32{{$}}
; CHECK-NEXT: i32.select $push0=, $1, $2, $0{{$}}
; CHECK-NEXT: return     $pop0{{$}}
define i32 @select_i32_bool(i1 zeroext %a, i32 %b, i32 %c) {
  %cond = select i1 %a, i32 %b, i32 %c
  ret i32 %cond
}

; CHECK-LABEL: select_i32_eq:
; CHECK-NEXT: .param     i32, i32, i32{{$}}
; CHECK-NEXT: .result    i32{{$}}
; CHECK-NEXT: i32.select $push0=, $2, $1, $0{{$}}
; CHECK-NEXT: return     $pop0{{$}}
define i32 @select_i32_eq(i32 %a, i32 %b, i32 %c) {
  %cmp = icmp eq i32 %a, 0
  %cond = select i1 %cmp, i32 %b, i32 %c
  ret i32 %cond
}

; CHECK-LABEL: select_i32_ne:
; CHECK-NEXT: .param     i32, i32, i32{{$}}
; CHECK-NEXT: .result    i32{{$}}
; CHECK-NEXT: i32.select $push0=, $1, $2, $0{{$}}
; CHECK-NEXT: return     $pop0{{$}}
define i32 @select_i32_ne(i32 %a, i32 %b, i32 %c) {
  %cmp = icmp ne i32 %a, 0
  %cond = select i1 %cmp, i32 %b, i32 %c
  ret i32 %cond
}

; CHECK-LABEL: select_i64_bool:
; CHECK-NEXT: .param     i32, i64, i64{{$}}
; CHECK-NEXT: .result    i64{{$}}
; CHECK-NEXT: i64.select $push0=, $1, $2, $0{{$}}
; CHECK-NEXT: return     $pop0{{$}}
define i64 @select_i64_bool(i1 zeroext %a, i64 %b, i64 %c) {
  %cond = select i1 %a, i64 %b, i64 %c
  ret i64 %cond
}

; CHECK-LABEL: select_i64_eq:
; CHECK-NEXT: .param     i32, i64, i64{{$}}
; CHECK-NEXT: .result    i64{{$}}
; CHECK-NEXT: i64.select $push0=, $2, $1, $0{{$}}
; CHECK-NEXT: return     $pop0{{$}}
define i64 @select_i64_eq(i32 %a, i64 %b, i64 %c) {
  %cmp = icmp eq i32 %a, 0
  %cond = select i1 %cmp, i64 %b, i64 %c
  ret i64 %cond
}

; CHECK-LABEL: select_i64_ne:
; CHECK-NEXT: .param     i32, i64, i64{{$}}
; CHECK-NEXT: .result    i64{{$}}
; CHECK-NEXT: i64.select $push0=, $1, $2, $0{{$}}
; CHECK-NEXT: return     $pop0{{$}}
define i64 @select_i64_ne(i32 %a, i64 %b, i64 %c) {
  %cmp = icmp ne i32 %a, 0
  %cond = select i1 %cmp, i64 %b, i64 %c
  ret i64 %cond
}

; CHECK-LABEL: select_f32_bool:
; CHECK-NEXT: .param     i32, f32, f32{{$}}
; CHECK-NEXT: .result    f32{{$}}
; CHECK-NEXT: f32.select $push0=, $1, $2, $0{{$}}
; CHECK-NEXT: return     $pop0{{$}}
define float @select_f32_bool(i1 zeroext %a, float %b, float %c) {
  %cond = select i1 %a, float %b, float %c
  ret float %cond
}

; CHECK-LABEL: select_f32_eq:
; CHECK-NEXT: .param     i32, f32, f32{{$}}
; CHECK-NEXT: .result    f32{{$}}
; CHECK-NEXT: f32.select $push0=, $2, $1, $0{{$}}
; CHECK-NEXT: return     $pop0{{$}}
define float @select_f32_eq(i32 %a, float %b, float %c) {
  %cmp = icmp eq i32 %a, 0
  %cond = select i1 %cmp, float %b, float %c
  ret float %cond
}

; CHECK-LABEL: select_f32_ne:
; CHECK-NEXT: .param     i32, f32, f32{{$}}
; CHECK-NEXT: .result    f32{{$}}
; CHECK-NEXT: f32.select $push0=, $1, $2, $0{{$}}
; CHECK-NEXT: return     $pop0{{$}}
define float @select_f32_ne(i32 %a, float %b, float %c) {
  %cmp = icmp ne i32 %a, 0
  %cond = select i1 %cmp, float %b, float %c
  ret float %cond
}

; CHECK-LABEL: select_f64_bool:
; CHECK-NEXT: .param     i32, f64, f64{{$}}
; CHECK-NEXT: .result    f64{{$}}
; CHECK-NEXT: f64.select $push0=, $1, $2, $0{{$}}
; CHECK-NEXT: return     $pop0{{$}}
define double @select_f64_bool(i1 zeroext %a, double %b, double %c) {
  %cond = select i1 %a, double %b, double %c
  ret double %cond
}

; CHECK-LABEL: select_f64_eq:
; CHECK-NEXT: .param     i32, f64, f64{{$}}
; CHECK-NEXT: .result    f64{{$}}
; CHECK-NEXT: f64.select $push0=, $2, $1, $0{{$}}
; CHECK-NEXT: return     $pop0{{$}}
define double @select_f64_eq(i32 %a, double %b, double %c) {
  %cmp = icmp eq i32 %a, 0
  %cond = select i1 %cmp, double %b, double %c
  ret double %cond
}

; CHECK-LABEL: select_f64_ne:
; CHECK-NEXT: .param     i32, f64, f64{{$}}
; CHECK-NEXT: .result    f64{{$}}
; CHECK-NEXT: f64.select $push0=, $1, $2, $0{{$}}
; CHECK-NEXT: return     $pop0{{$}}
define double @select_f64_ne(i32 %a, double %b, double %c) {
  %cmp = icmp ne i32 %a, 0
  %cond = select i1 %cmp, double %b, double %c
  ret double %cond
}
