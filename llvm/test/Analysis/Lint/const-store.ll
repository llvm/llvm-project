; RUN: not opt --mtriple=amdgcn --passes=lint --lint-abort-on-error %s -o /dev/null |& FileCheck %s
; RUN: opt --mtriple=amdgcn --mcpu=gfx1030 --passes=lint %s -o /dev/null |& FileCheck %s --check-prefixes=CHECK,CHECK0
; RUN: opt --mtriple=x86_64 --passes=lint --lint-abort-on-error %s -o /dev/null |& FileCheck %s --allow-empty --check-prefix=NOERR
; NOERR: {{^$}}

define amdgpu_kernel void @store_const(ptr addrspace(4) %out, i32 %a, i32 %b) {
; CHECK: Undefined behavior: Write to const memory
; CHECK-NEXT: store i32 %r, ptr addrspace(4) %out
  %r = add i32 %a, %b
  store i32 %r, ptr addrspace(4) %out
  ret void
}

define amdgpu_kernel void @memcpy_to_const(ptr addrspace(6) %dst, ptr %src) {
; CHECK0: Undefined behavior: Write to const memory
; CHECK0-NEXT: call void @llvm.memcpy.p6.p0.i32(ptr addrspace(6) %dst, ptr %src, i32 256, i1 false)
  call void @llvm.memcpy.px.py.i32(ptr addrspace(6) %dst, ptr %src, i32 256, i1 false)
  ret void
}

declare void @llvm.memcpy.px.py.i32(ptr addrspace(6) noalias nocapture writeonly, ptr noalias nocapture readonly, i32, i1 immarg)
