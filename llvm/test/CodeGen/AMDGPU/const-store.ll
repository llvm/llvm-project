; RUN: not llc -mtriple=amdgcn < %s |& FileCheck %s

define amdgpu_kernel void @store_const(ptr addrspace(4) %out, i32 %a, i32 %b) {
; CHECK: Store cannot be to constant addrspace
; CHECK-NEXT: store i32 %r, ptr addrspace(4) %out
  %r = add i32 %a, %b
  store i32 %r, ptr addrspace(4) %out
  ret void
}
