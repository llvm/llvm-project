; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare i32 @llvm.amdgcn.cvt.pk.u8.f32(float, i32, i32) #0

; GCN-LABEL: {{^}}v_cvt_pk_u8_f32_idx_0:
; GCN: v_cvt_pk_u8_f32 v{{[0-9]+}}, s{{[0-9]+}}, 0, v{{[0-9]+}}
define amdgpu_kernel void @v_cvt_pk_u8_f32_idx_0(ptr addrspace(1) %out, float %src, i32 %reg) {
  %result = call i32 @llvm.amdgcn.cvt.pk.u8.f32(float %src, i32 0, i32 %reg) #0
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_cvt_pk_u8_f32_idx_1:
; GCN: v_cvt_pk_u8_f32 v{{[0-9]+}}, s{{[0-9]+}}, 1, v{{[0-9]+}}
define amdgpu_kernel void @v_cvt_pk_u8_f32_idx_1(ptr addrspace(1) %out, float %src, i32 %reg) {
  %result = call i32 @llvm.amdgcn.cvt.pk.u8.f32(float %src, i32 1, i32 %reg) #0
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_cvt_pk_u8_f32_idx_2:
; GCN: v_cvt_pk_u8_f32 v{{[0-9]+}}, s{{[0-9]+}}, 2, v{{[0-9]+}}
define amdgpu_kernel void @v_cvt_pk_u8_f32_idx_2(ptr addrspace(1) %out, float %src, i32 %reg) {
  %result = call i32 @llvm.amdgcn.cvt.pk.u8.f32(float %src, i32 2, i32 %reg) #0
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_cvt_pk_u8_f32_idx_3:
; GCN: v_cvt_pk_u8_f32 v{{[0-9]+}}, s{{[0-9]+}}, 3, v{{[0-9]+}}
define amdgpu_kernel void @v_cvt_pk_u8_f32_idx_3(ptr addrspace(1) %out, float %src, i32 %reg) {
  %result = call i32 @llvm.amdgcn.cvt.pk.u8.f32(float %src, i32 3, i32 %reg) #0
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_cvt_pk_u8_f32_combine:
; GCN: v_cvt_pk_u8_f32 v{{[0-9]+}}, s{{[0-9]+}}, 0, v{{[0-9]+}}
; GCN: v_cvt_pk_u8_f32 v{{[0-9]+}}, s{{[0-9]+}}, 1, v{{[0-9]+}}
; GCN: v_cvt_pk_u8_f32 v{{[0-9]+}}, s{{[0-9]+}}, 2, v{{[0-9]+}}
; GCN: v_cvt_pk_u8_f32 v{{[0-9]+}}, s{{[0-9]+}}, 3, v{{[0-9]+}}
define amdgpu_kernel void @v_cvt_pk_u8_f32_combine(ptr addrspace(1) %out, float %src, i32 %reg) {
  %result0 = call i32 @llvm.amdgcn.cvt.pk.u8.f32(float %src, i32 0, i32 %reg) #0
  %result1 = call i32 @llvm.amdgcn.cvt.pk.u8.f32(float %src, i32 1, i32 %result0) #0
  %result2 = call i32 @llvm.amdgcn.cvt.pk.u8.f32(float %src, i32 2, i32 %result1) #0
  %result3 = call i32 @llvm.amdgcn.cvt.pk.u8.f32(float %src, i32 3, i32 %result2) #0
  store i32 %result3, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_cvt_pk_u8_f32_idx:
; GCN: v_cvt_pk_u8_f32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @v_cvt_pk_u8_f32_idx(ptr addrspace(1) %out, float %src, i32 %idx, i32 %reg) {
  %result = call i32 @llvm.amdgcn.cvt.pk.u8.f32(float %src, i32 %idx, i32 %reg) #0
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

attributes #0 = { nounwind readnone }
