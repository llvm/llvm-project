; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=null %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -filetype=null %s 2>&1 | FileCheck %s

; The AMDGPU backend does not support f64 FLOG/FLOG2/FLOG10 lowering.
; These fall through to a libcall that cannot be formed.  Previously
; this silently produced a store of zero/poison; ensure it is now a
; hard error.

; CHECK: LLVM ERROR: unsupported libcall legalization

declare double @llvm.log2.f64(double)

define amdgpu_kernel void @log2_f64(ptr addrspace(1) %out, double %x) {
  %r = call double @llvm.log2.f64(double %x)
  store double %r, ptr addrspace(1) %out
  ret void
}
