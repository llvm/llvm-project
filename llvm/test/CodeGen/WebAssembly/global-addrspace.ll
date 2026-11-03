; RUN: llc < %s -mtriple=wasm32-unknown-unknown | FileCheck %s

;; This test ensures that globals in addrspace(1) (Wasm-specific variables)
;; do not cause a crash during emission when Subtarget is not set
;; (e.g., in modules without functions) and correctly emit their initializers.

; CHECK: .globaltype wasm_var, i32
; CHECK-NEXT: .globalinit wasm_var, i32.const 42
; CHECK: .globl wasm_var
; CHECK-LABEL: wasm_var:
@wasm_var = addrspace(1) global i32 42

; CHECK: .globaltype wasm_var_float, f32
; CHECK-NEXT: .globalinit wasm_var_float, f32.const 0x1.91eb86p1
; CHECK: .globl wasm_var_float
; CHECK-LABEL: wasm_var_float:
@wasm_var_float = addrspace(1) global float 0x40091EB860000000

; CHECK: .globaltype     wasm_var_i64, i64
; CHECK-NEXT: .globalinit wasm_var_i64, i64.const 1234567890
; CHECK: .globl  wasm_var_i64
; CHECK-LABEL: wasm_var_i64:
@wasm_var_i64 = addrspace(1) global i64 1234567890

; CHECK: .globaltype     wasm_var_f64, f64
; CHECK-NEXT: .globalinit wasm_var_f64, f64.const -0x0p0
; CHECK: .globl  wasm_var_f64
; CHECK-LABEL: wasm_var_f64:
@wasm_var_f64 = local_unnamed_addr addrspace(1) global double -0.0

; CHECK: .globaltype wasm_external, i32
; CHECK-NOT: .global wasm_external
; CHECK-NOT: .globalinit wasm_external
; CHECK-NOT: wasm_external:
@wasm_external = external addrspace(1) global i32
