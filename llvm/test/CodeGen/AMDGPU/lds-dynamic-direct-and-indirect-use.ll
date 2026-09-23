; RUN: llc -mtriple=amdgpu9.42-amd-amdhsa < %s | FileCheck %s
; RUN: llc -mtriple=amdgpu9.42-amd-amdhsa -global-isel < %s | FileCheck %s

; A kernel that accesses a dynamic LDS variable both directly and through a
; non-inlined callee. amdgpu-lower-module-lds rewrites only the callee's uses
; to go through llvm.amdgcn.dynlds.offset.table and gives the per-kernel
; llvm.amdgcn.kernel.dynlds variable absolute_symbol metadata. The kernel's
; direct use still refers to the original @dynlds, which has no such metadata.

@dynlds = external addrspace(3) global [0 x i8], align 16

; CHECK-LABEL: helper:
; CHECK: llvm.amdgcn.dynlds.offset.table@rel32@lo
; CHECK: ds_write_b8
define internal void @helper() noinline {
  store volatile i8 1, ptr addrspace(3) @dynlds, align 1
  ret void
}

; CHECK-LABEL: kernel:
; CHECK: ds_write_b8 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; CHECK: s_swappc_b64
define amdgpu_kernel void @kernel() {
  %p = getelementptr i8, ptr addrspace(3) @dynlds, i32 16
  store volatile i8 2, ptr addrspace(3) %p, align 1
  call void @helper()
  ret void
}
