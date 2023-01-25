; RUN: opt -mtriple amdgcn-amdhsa -mcpu=gfx90a -passes=legacy-divergence-analysis < %s -S 2>&1 | FileCheck -check-prefix=OPT %s
; RUN: llc -mtriple amdgcn-amdhsa -mcpu=fiji -amdgpu-scalarize-global-loads -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.readfirstlane(i32)

; OPT-LABEL: define amdgpu_kernel void @readfirstlane_uniform(
; OPT-NEXT:    %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
; OPT-NEXT:    %scalar = tail call i32 @llvm.amdgcn.readfirstlane(i32 %tid)
; OPT-NEXT:    %idx = zext i32 %scalar to i64
; OPT-NEXT:    %gep0 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %idx
; OPT-NEXT:    %val = load float, ptr addrspace(1) %gep0, align 4
; OPT-NEXT:    %gep1 = getelementptr inbounds float, ptr addrspace(1) %1, i64 10
; OPT-NEXT:    store float %val, ptr addrspace(1) %gep1, align 4
; OPT-NEXT:    ret void
;
; GCN-LABEL: readfirstlane_uniform
; GCN: 	s_load_dwordx4 s[[[IN_ADDR:[0-9]+]]:3], s[4:5], 0x0
; GCN:  v_readfirstlane_b32 s[[SCALAR:[0-9]+]], v0
; GCN: 	s_add_u32 s[[LOAD_ADDR:[0-9]+]], s[[IN_ADDR]], s[[SCALAR]]
; GCN:	s_load_dword s{{[0-9]+}}, s[[[LOAD_ADDR]]

define amdgpu_kernel void @readfirstlane_uniform(ptr addrspace(1) noalias nocapture readonly, ptr addrspace(1) noalias nocapture readonly) {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %scalar = tail call i32 @llvm.amdgcn.readfirstlane(i32 %tid)
  %idx = zext i32 %scalar to i64
  %gep0 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %idx
  %val = load float, ptr addrspace(1) %gep0, align 4
  %gep1 = getelementptr inbounds float, ptr addrspace(1) %1, i64 10
  store float %val, ptr addrspace(1) %gep1, align 4
  ret void
}
