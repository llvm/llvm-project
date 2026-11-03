; RUN: llc < %s -O2 -mtriple=wasm32-unknown-unknown | FileCheck %s

; Do not fold a select of wasm_var loads into a load through a select of their
; addresses. WebAssembly globals are accessed by name, not computed addresses.

@g1 = external addrspace(1) global i32
@g2 = external addrspace(1) global i32

define i32 @select_global_load(i1 %cond) {
; CHECK-LABEL: select_global_load:
; CHECK:       global.get g1
; CHECK:       global.get g2
; CHECK:       i32.select
  %g1_value = load i32, ptr addrspace(1) @g1
  %g2_value = load i32, ptr addrspace(1) @g2
  %result = select i1 %cond, i32 %g1_value, i32 %g2_value
  ret i32 %result
}

define i32 @selectcc_global_load(i32 %cond) {
; CHECK-LABEL: selectcc_global_load:
; CHECK:       global.get g1
; CHECK:       global.get g2
; CHECK:       i32.select
  %nonzero = icmp ne i32 %cond, 0
  %g1_value = load i32, ptr addrspace(1) @g1
  %g2_value = load i32, ptr addrspace(1) @g2
  %result = select i1 %nonzero, i32 %g1_value, i32 %g2_value
  ret i32 %result
}
