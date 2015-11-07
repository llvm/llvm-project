; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic 32-bit floating-point comparison operations assemble as
; expected.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: ord_f32:
; CHECK-NEXT: .param f32{{$}}
; CHECK-NEXT: .param f32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: .local f32, f32, i32, i32, i32{{$}}
; CHECK-NEXT: get_local push, 0{{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: get_local push, 1{{$}}
; CHECK-NEXT: set_local 3, pop{{$}}
; CHECK-NEXT: f32.eq push, (get_local 3), (get_local 3){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
; CHECK-NEXT: f32.eq push, (get_local 2), (get_local 2){{$}}
; CHECK-NEXT: set_local 5, pop{{$}}
; CHECK-NEXT: i32.and push, (get_local 5), (get_local 4){{$}}
; CHECK-NEXT: set_local 6, pop{{$}}
; CHECK-NEXT: return (get_local 6){{$}}
define i32 @ord_f32(float %x, float %y) {
  %a = fcmp ord float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: uno_f32:
; CHECK-NEXT: .param f32{{$}}
; CHECK-NEXT: .param f32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: .local f32, f32, i32, i32, i32{{$}}
; CHECK-NEXT: get_local push, 0{{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: get_local push, 1{{$}}
; CHECK-NEXT: set_local 3, pop{{$}}
; CHECK-NEXT: f32.ne push, (get_local 3), (get_local 3){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
; CHECK-NEXT: f32.ne push, (get_local 2), (get_local 2){{$}}
; CHECK-NEXT: set_local 5, pop{{$}}
; CHECK-NEXT: i32.or push, (get_local 5), (get_local 4){{$}}
; CHECK-NEXT: set_local 6, pop{{$}}
; CHECK-NEXT: return (get_local 6){{$}}
define i32 @uno_f32(float %x, float %y) {
  %a = fcmp uno float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: oeq_f32:
; CHECK-NEXT: .param f32{{$}}
; CHECK-NEXT: .param f32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: .local f32, f32, i32{{$}}
; CHECK-NEXT: get_local push, 1{{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: get_local push, 0{{$}}
; CHECK-NEXT: set_local 3, pop{{$}}
; CHECK-NEXT: f32.eq push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
; CHECK-NEXT: return (get_local 4){{$}}
define i32 @oeq_f32(float %x, float %y) {
  %a = fcmp oeq float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: une_f32:
; CHECK: f32.ne push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
define i32 @une_f32(float %x, float %y) {
  %a = fcmp une float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: olt_f32:
; CHECK: f32.lt push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
define i32 @olt_f32(float %x, float %y) {
  %a = fcmp olt float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ole_f32:
; CHECK: f32.le push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
define i32 @ole_f32(float %x, float %y) {
  %a = fcmp ole float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ogt_f32:
; CHECK: f32.gt push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
define i32 @ogt_f32(float %x, float %y) {
  %a = fcmp ogt float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: oge_f32:
; CHECK: f32.ge push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
define i32 @oge_f32(float %x, float %y) {
  %a = fcmp oge float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; Expanded comparisons, which also check for NaN.

; CHECK-LABEL: ueq_f32:
; CHECK-NEXT: .param f32{{$}}
; CHECK-NEXT: .param f32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: .local f32, f32, i32, i32, i32, i32, i32{{$}}
; CHECK-NEXT: get_local push, 1{{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: get_local push, 0{{$}}
; CHECK-NEXT: set_local 3, pop{{$}}
; CHECK-NEXT: f32.eq push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
; CHECK-NEXT: f32.ne push, (get_local 2), (get_local 2){{$}}
; CHECK-NEXT: set_local 5, pop{{$}}
; CHECK-NEXT: f32.ne push, (get_local 3), (get_local 3){{$}}
; CHECK-NEXT: set_local 6, pop{{$}}
; CHECK-NEXT: i32.or push, (get_local 6), (get_local 5){{$}}
; CHECK-NEXT: set_local 7, pop{{$}}
; CHECK-NEXT: i32.or push, (get_local 4), (get_local 7){{$}}
; CHECK-NEXT: set_local 8, pop{{$}}
; CHECK-NEXT: return (get_local 8){{$}}
define i32 @ueq_f32(float %x, float %y) {
  %a = fcmp ueq float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: one_f32:
; CHECK: f32.ne push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
define i32 @one_f32(float %x, float %y) {
  %a = fcmp one float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ult_f32:
; CHECK: f32.lt push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
define i32 @ult_f32(float %x, float %y) {
  %a = fcmp ult float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ule_f32:
; CHECK: f32.le push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
define i32 @ule_f32(float %x, float %y) {
  %a = fcmp ule float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ugt_f32:
; CHECK: f32.gt push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
define i32 @ugt_f32(float %x, float %y) {
  %a = fcmp ugt float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: uge_f32:
; CHECK: f32.ge push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
define i32 @uge_f32(float %x, float %y) {
  %a = fcmp uge float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}
