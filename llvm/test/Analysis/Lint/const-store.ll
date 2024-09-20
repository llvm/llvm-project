; RUN: not opt --mtriple=amdgcn --passes=lint --lint-abort-on-error %s -o - |& FileCheck %s
; RUN: not opt --mtriple=amdgcn --mcpu=gfx1030 --passes=lint --lint-abort-on-error %s -o - |& FileCheck %s

define amdgpu_kernel void @store_const(ptr addrspace(4) %out, i32 %a, i32 %b) {
; CHECK: Undefined behavior: Write to const memory
; CHECK-NEXT: store i32 %r, ptr addrspace(4) %out
  %r = add i32 %a, %b
  store i32 %r, ptr addrspace(4) %out
  ret void
}
