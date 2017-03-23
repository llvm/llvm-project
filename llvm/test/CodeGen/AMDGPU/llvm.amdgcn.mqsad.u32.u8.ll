; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare <4 x i32> @llvm.amdgcn.mqsad.u32.u8(i64, i32, <4 x i32>) #0

; GCN-LABEL: {{^}}v_mqsad_u32_u8_use_non_inline_constant:
; GCN: v_mqsad_u32_u8 v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
define amdgpu_kernel void @v_mqsad_u32_u8_use_non_inline_constant(<4 x i32> addrspace(1)* %out, i64 %src) {
  %result = call <4 x i32> @llvm.amdgcn.mqsad.u32.u8(i64 %src, i32 100, <4 x i32> <i32 100, i32 100, i32 100, i32 100>) #0
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_mqsad_u32_u8_non_immediate:
; GCN: v_mqsad_u32_u8 v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
define amdgpu_kernel void @v_mqsad_u32_u8_non_immediate(<4 x i32> addrspace(1)* %out, i64 %src, i32 %a, <4 x i32> %b) {
  %result = call <4 x i32> @llvm.amdgcn.mqsad.u32.u8(i64 %src, i32 %a, <4 x i32> %b) #0
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_mqsad_u32_u8_inline_integer_immediate:
; GCN: v_mqsad_u32_u8 v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
define amdgpu_kernel void @v_mqsad_u32_u8_inline_integer_immediate(<4 x i32> addrspace(1)* %out, i64 %src, i32 %a) {
  %result = call <4 x i32> @llvm.amdgcn.mqsad.u32.u8(i64 %src, i32 %a, <4 x i32> <i32 10, i32 20, i32 30, i32 40>) #0
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_mqsad_u32_u8_inline_fp_immediate:
; GCN: v_mqsad_u32_u8 v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
define amdgpu_kernel void @v_mqsad_u32_u8_inline_fp_immediate(<4 x i32> addrspace(1)* %out, i64 %src, i32 %a) {
  %result = call <4 x i32> @llvm.amdgcn.mqsad.u32.u8(i64 %src, i32 %a, <4 x i32> <i32 1065353216, i32 0, i32 0, i32 0>) #0
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_mqsad_u32_u8_use_sgpr_vgpr:
; GCN: v_mqsad_u32_u8 v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
define amdgpu_kernel void @v_mqsad_u32_u8_use_sgpr_vgpr(<4 x i32> addrspace(1)* %out, i64 %src, i32 %a, <4 x i32> addrspace(1)* %input) {
  %in = load <4 x i32>, <4 x i32> addrspace(1) * %input

  %result = call <4 x i32> @llvm.amdgcn.mqsad.u32.u8(i64 %src, i32 %a, <4 x i32> %in) #0
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out, align 4
  ret void
}

attributes #0 = { nounwind readnone }
