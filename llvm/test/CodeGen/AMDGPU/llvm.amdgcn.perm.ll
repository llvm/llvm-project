; RUN: llc -mtriple=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -global-isel -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare i32 @llvm.amdgcn.perm(i32, i32, i32) #0

; GCN-LABEL: {{^}}v_perm_b32_v_v_v:
; GCN: v_perm_b32 v{{[0-9]+}}, v0, v1, v2
define amdgpu_ps void @v_perm_b32_v_v_v(i32 %src1, i32 %src2, i32 %src3, ptr addrspace(1) %out) #1 {
  %val = call i32 @llvm.amdgcn.perm(i32 %src1, i32 %src2, i32 %src3) #0
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_perm_b32_v_v_c:
; GCN: v_perm_b32 v{{[0-9]+}}, v0, v1, {{[vs][0-9]+}}
define amdgpu_ps void @v_perm_b32_v_v_c(i32 %src1, i32 %src2, ptr addrspace(1) %out) #1 {
  %val = call i32 @llvm.amdgcn.perm(i32 %src1, i32 %src2, i32 12345) #0
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_perm_b32_s_v_c:
; GCN: v_perm_b32 v{{[0-9]+}}, s0, v0, v{{[0-9]+}}
define amdgpu_ps void @v_perm_b32_s_v_c(i32 inreg %src1, i32 %src2, ptr addrspace(1) %out) #1 {
  %val = call i32 @llvm.amdgcn.perm(i32 %src1, i32 %src2, i32 12345) #0
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_perm_b32_s_s_c:
; GCN: v_perm_b32 v{{[0-9]+}}, s0, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_ps void @v_perm_b32_s_s_c(i32 inreg %src1, i32 inreg %src2, ptr addrspace(1) %out) #1 {
  %val = call i32 @llvm.amdgcn.perm(i32 %src1, i32 %src2, i32 12345) #0
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_perm_b32_v_s_i:
; GCN: v_perm_b32 v{{[0-9]+}}, v0, s0, 1
define amdgpu_ps void @v_perm_b32_v_s_i(i32 %src1, i32 inreg %src2, ptr addrspace(1) %out) #1 {
  %val = call i32 @llvm.amdgcn.perm(i32 %src1, i32 %src2, i32 1) #0
  store i32 %val, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
