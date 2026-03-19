; REQUIRES: amdgpu-registered-target
; RUN: not opt -S -mtriple=amdgcn-amd-amdhsa -passes=hipstdpar-select-accelerator-code \
; RUN: %s 2>&1 | FileCheck %s

; CHECK: error: The Indirection Table must have 3 elements; 2 is incorrect.
%class.anon = type { i64, ptr }
@a = external hidden local_unnamed_addr addrspace(1) global ptr, align 8
@__hipstdpar_symbol_indirection_table = weak_odr protected addrspace(4) externally_initialized constant %class.anon zeroinitializer, align 8

define amdgpu_kernel void @store(ptr %p) {
entry:
  store ptr %p, ptr addrspace(1) @a, align 8
  ret void
}
