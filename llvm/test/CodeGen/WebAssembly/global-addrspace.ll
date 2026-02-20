; RUN: llc < %s -mtriple=wasm32-unknown-unknown | FileCheck %s

;; This test ensures that globals in addrspace(1) (Wasm-specific variables)
;; do not cause a crash during emission when Subtarget is not set
;; (e.g., in modules without functions) and correctly emit their initializers.

; CHECK-LABEL: .globaltype wasm_var, i32
; CHECK-NEXT: .globl wasm_var
; CHECK-NEXT: wasm_var:
@wasm_var = addrspace(1) global i32 42

; CHECK-LABEL: .globaltype wasm_var_float, f32
; CHECK-NEXT: .globl wasm_var_float
; CHECK-NEXT: wasm_var_float:
@wasm_var_float = addrspace(1) global float 0x40091EB860000000

; CHECK-LABEL: .globaltype     wasm_var_i64, i64
; CHECK-NEXT: .globl  wasm_var_i64
; CHECK-NEXT: wasm_var_i64:
@wasm_var_i64 = addrspace(1) global i64 1234567890
