; RUN: llc -mcpu=gfx1200 < %s | FileCheck %s --check-prefixes=CHECK,PACKED
; RUN: llc -mcpu=gfx1030 < %s | FileCheck %s --check-prefixes=CHECK,NOTPACKED
target triple = "amdgcn-amd-amdhsa"

@global = addrspace(1) global i32 poison, align 4

; Carefully crafted kernel that uses v0 but never writes a VGPR or reads another VGPR.
; Only hardware-initialized VGPRs (v0) are read in this kernel.

; CHECK-LABEL: amdhsa.kernels:
; CHECK-LABEL: kernel_x
; CHECK: .vgpr_count:     1
define amdgpu_kernel void @kernel_x(ptr addrspace(8) %rsrc) #0 {
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %id, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0)
  ret void
}

; CHECK-LABEL: kernel_z
; PACKED: .vgpr_count:     1
; NOTPACKED: .vgpr_count:     3
define amdgpu_kernel void @kernel_z(ptr addrspace(8) %rsrc) {
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.z()
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %id, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0)
  ret void
}

attributes #0 = { "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" }
