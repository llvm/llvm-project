; RUN: not opt --mtriple=amdgcn --passes=lint --lint-abort-on-error %s -disable-output 2>&1 | FileCheck %s
; RUN: opt --mtriple=amdgcn --mcpu=gfx1030 --passes=lint %s -disable-output 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK0
; RUN: opt --mtriple=x86_64 --passes=lint --lint-abort-on-error %s -disable-output 2>&1 | FileCheck %s --allow-empty --check-prefix=NOERR
; NOERR: {{^$}}

define amdgpu_kernel void @store_const(ptr addrspace(4) %out, i32 %a, i32 %b) {
; CHECK: Undefined behavior: Write to memory in const addrspace
; CHECK-NEXT: store i32 %r, ptr addrspace(4) %out
  %r = add i32 %a, %b
  store i32 %r, ptr addrspace(4) %out
  ret void
}

declare void @llvm.memset.p4.i64(ptr addrspace(4) noalias nocapture writeonly, i8, i64, i1)
define amdgpu_kernel void @memset_const(ptr addrspace(4) %dst) {
; CHECK0: Undefined behavior: Write to memory in const addrspace
; CHECK0-NEXT: call void @llvm.memset.p4.i64(ptr addrspace(4) %dst, i8 0, i64 256, i1 false)
  call void @llvm.memset.p4.i64(ptr addrspace(4) %dst, i8 0, i64 256, i1 false)
  ret void
}

declare void @llvm.memcpy.p6.p0.i32(ptr addrspace(6) noalias nocapture writeonly, ptr noalias nocapture readonly, i32, i1)
define amdgpu_kernel void @memcpy_to_const(ptr addrspace(6) %dst, ptr %src) {
; CHECK0: Undefined behavior: Write to memory in const addrspace
; CHECK0-NEXT: call void @llvm.memcpy.p6.p0.i32(ptr addrspace(6) %dst, ptr %src, i32 256, i1 false)
  call void @llvm.memcpy.p6.p0.i32(ptr addrspace(6) %dst, ptr %src, i32 256, i1 false)
  ret void
}

define amdgpu_kernel void @cmpxchg_to_const(ptr addrspace(4) %dst, i32 %src) {
; CHECK0: Undefined behavior: Write to memory in const addrspace
; CHECK0-NEXT: %void = cmpxchg ptr addrspace(4) %dst, i32 0, i32 %src seq_cst monotonic
  %void = cmpxchg ptr addrspace(4) %dst, i32 0, i32 %src seq_cst monotonic
  ret void
}

define amdgpu_kernel void @atomicrmw_to_const(ptr addrspace(4) %dst, i32 %src) {
; CHECK0: Undefined behavior: Write to memory in const addrspace
; CHECK0-NEXT: %void = atomicrmw add ptr addrspace(4) %dst, i32 %src acquire
  %void = atomicrmw add ptr addrspace(4) %dst, i32 %src acquire
  ret void
}

declare void @const_param(ptr addrspace(6))
define amdgpu_kernel void @call_with_const(ptr addrspace(6) %dst) {
; CHECK0-NOT: call void @const_param(ptr addrspace(6) %dst)
  call void @const_param(ptr addrspace(6) %dst)
  ret void
}
