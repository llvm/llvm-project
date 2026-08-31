; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa -amdgpu-enable-object-linking < %s | FileCheck %s

; Object-linking lowering externalizes LDS definitions. Verify that it also
; removes COMDAT membership, which is invalid on a declaration.

$lds = comdat any

@lds = linkonce_odr addrspace(3) global [4 x i32] undef, comdat($lds), align 16

; CHECK: .amdgpu_info kernel
; CHECK: .amdgpu_use lds
; CHECK: .amdgpu_lds lds, 16, 16

define amdgpu_kernel void @kernel() {
  %lds.ptr = getelementptr [4 x i32], ptr addrspace(3) @lds, i32 0, i32 0
  store volatile i32 1, ptr addrspace(3) %lds.ptr, align 16
  ret void
}
