; Runs original SDAG test with -global-isel

; RUN: not llc -global-isel -mtriple=amdgcn-amd-amdhsa -o /dev/null < %S/../lds-global-non-entry-func.ll 2>&1 | FileCheck %s

@lds = internal addrspace(3) global float undef, align 4

; CHECK: error: <unknown>:0:0: in function func_use_lds_global void (): local memory global used by non-kernel function
; CHECK-NOT: error
; CHECK-NOT: ERROR
define void @func_use_lds_global() {
  store float 0.0, float addrspace(3)* @lds, align 4
  ret void
}
