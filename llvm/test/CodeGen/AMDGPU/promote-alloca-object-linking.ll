; RUN: llc -mtriple=amdgpu9.42-amd-amdhsa -mcpu=gfx942 -amdgpu-enable-object-linking < %s | FileCheck %s
; RUN: llc -enable-new-pm -mtriple=amdgpu9.42-amd-amdhsa -mcpu=gfx942 -amdgpu-enable-object-linking < %s | FileCheck %s

; The second object-linking LDS lowering run must process LDS globals created by
; PromoteAlloca.

; CHECK: .amdgpu_info kernel
; CHECK: .amdgpu_use kernel.stack
; CHECK: .amdgpu_lds kernel.stack, 4096, 4

declare ptr @llvm.invariant.start.p5(i64, ptr addrspace(5) nocapture)
declare void @llvm.invariant.end.p5(ptr, i64, ptr addrspace(5) nocapture)

define amdgpu_kernel void @kernel() {
  %stack = alloca i32, align 4, addrspace(5)
  %invariant = call ptr @llvm.invariant.start.p5(i64 0,
                                                 ptr addrspace(5) %stack)
  store <2 x i1> zeroinitializer, ptr %invariant, align 1
  call void @llvm.invariant.end.p5(ptr %invariant, i64 0,
                                   ptr addrspace(5) %stack)
  ret void
}
