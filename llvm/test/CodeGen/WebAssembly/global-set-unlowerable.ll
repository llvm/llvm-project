; RUN: not llc < %s --mtriple=wasm32 2>&1 | FileCheck %s

; Demonstrates a code pattern that could be encountered even with frontend
; restrictions on creating new pointers to globals. In the absence of a better
; fix, the backend should produce a comprehensible message for why it can't
; continue.

; CHECK: LVM ERROR: Encountered an unlowerable store to the wasm_var address space

@g1 = external addrspace(1) global i32
@g2 = external addrspace(1) global i32

define void @global_set_phi(i1 zeroext %bool) {
  %sel = select i1 %bool, ptr addrspace(1) @g1, ptr addrspace(1) @g2
  store i32 100, ptr addrspace(1) %sel
  ret void
}
