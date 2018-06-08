; RUN: not llc -mtriple=amdgcn-amd-amdhsa -o /dev/null %s 2>&1 | FileCheck %s

@lds = internal addrspace(3) global float undef, align 4

; CHECK: error: <unknown>:0:0: in function func_use_lds_global void (): local memory global used by non-kernel function
define void @func_use_lds_global() {
  store float 0.0, float addrspace(3)* @lds, align 4
  ret void
}
