; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false | FileCheck %s

@i32_global = local_unnamed_addr addrspace(1) global i32 undef
@i64_global = local_unnamed_addr addrspace(1) global i64 undef
@f32_global = local_unnamed_addr addrspace(1) global float undef
@f64_global = local_unnamed_addr addrspace(1) global double undef

@i32_external_used = external addrspace(1) global i32
@i32_external_unused = external addrspace(1) global i32

define i32 @return_i32_global() {
; CHECK-LABEL: return_i32_global:
; CHECK-NEXT: functype       return_i32_global () -> (i32)
; CHECK-NEXT: global.get i32_global
; CHECK-NEXT: end_function
  %v = load i32, i32 addrspace(1)* @i32_global
  ret i32 %v
}

define i64 @return_i64_global() {
; CHECK-LABEL: return_i64_global:
; CHECK-NEXT: functype       return_i64_global () -> (i64)
; CHECK-NEXT: global.get i64_global
; CHECK-NEXT: end_function
  %v = load i64, i64 addrspace(1)* @i64_global
  ret i64 %v
}

define float @return_f32_global() {
; CHECK-LABEL: return_f32_global:
; CHECK-NEXT: functype       return_f32_global () -> (f32)
; CHECK-NEXT: global.get f32_global
; CHECK-NEXT: end_function
  %v = load float, float addrspace(1)* @f32_global
  ret float %v
}

define double @return_f64_global() {
; CHECK-LABEL: return_f64_global:
; CHECK-NEXT: functype       return_f64_global () -> (f64)
; CHECK-NEXT: global.get f64_global
; CHECK-NEXT: end_function
  %v = load double, double addrspace(1)* @f64_global
  ret double %v
}

define i32 @return_extern_i32_global() {
; CHECK-LABEL: return_extern_i32_global:
; CHECK-NEXT: functype       return_extern_i32_global () -> (i32)
; CHECK-NEXT: global.get i32_external_used
; CHECK-NEXT: end_function
  %v = load i32, i32 addrspace(1)* @i32_external_used
  ret i32 %v
}


; CHECK: .globl i32_global
; CHECK: .globaltype i32_global, i32
; CHECK-LABEL: i32_global:

; CHECK: .globl i64_global
; CHECK: .globaltype i64_global, i64
; CHECK-LABEL: i64_global:

; CHECK: .globl f32_global
; CHECK: .globaltype f32_global, f32
; CHECK-LABEL: f32_global:

; CHECK: .globl f64_global
; CHECK: .globaltype f64_global, f64
; CHECK-LABEL: f64_global:

; CHECK-NOT: .global i32_external_used
; CHECK: .globaltype i32_external_used, i32
; CHECK-NOT: i32_external_used:

; CHECK-NOT: .global i32_external_unused
; CHECK: .globaltype i32_external_unused, i32
; CHECK-NOT: i32_external_unused:
