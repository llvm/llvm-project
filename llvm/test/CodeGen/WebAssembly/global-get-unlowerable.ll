; RUN: not llc < %s --mtriple=wasm32 2>&1 | FileCheck %s

; Demonstrates a code pattern that could be encountered even with frontend
; restrictions on creating new pointers to globals. In the absence of a better
; fix, the backend should produce a comprehensible message for why it can't
; continue.

; CHECK: LVM ERROR: Encountered an unlowerable load from the wasm_var address space

@g1 = external addrspace(1) global i32
@g2 = external addrspace(1) global i32

define i32 @global_get_phi(i1 zeroext %bool) {
  %sel = select i1 %bool, ptr addrspace(1) @g1, ptr addrspace(1) @g2
  %gval = load i32, ptr addrspace(1) %sel
  ret i32 %gval
}
