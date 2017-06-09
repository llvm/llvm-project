; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare <4 x i32> @llvm.amdgcn.mqsad.u32.u8(i64, i32, <4 x i32>) #0

; GCN-LABEL: {{^}}v_mqsad_u32_u8_inline_integer_immediate:
; GCN-DAG: v_mov_b32_e32 v0, v2
; GCN-DAG: v_mov_b32_e32 v1, v3
; GCN: v_mqsad_u32_u8 v[2:5], v[0:1], v6, v[{{[0-9]+:[0-9]+}}]
define amdgpu_kernel void @v_mqsad_u32_u8_inline_integer_immediate(<4 x i32> addrspace(1)* %out, i64 %src, i32 %a) {
  %tmp = call i64 asm "v_lsrlrev_b64 $0, $1, 1", "={v[2:3]},v"(i64 %src) #0
  %tmp1 = call i32 asm "v_mov_b32 $0, $1", "={v4},v"(i32 %a) #0
  %tmp2 = call <4 x i32> @llvm.amdgcn.mqsad.u32.u8(i64 %tmp, i32 %tmp1, <4 x i32> <i32 10, i32 20, i32 30, i32 40>) #0
  %tmp3 = call <4 x i32>  asm ";; force constraint", "=v,{v[2:5]}"(<4 x i32> %tmp2) #0
  store <4 x i32> %tmp3, <4 x i32> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_mqsad_u32_u8_non_immediate:
; GCN-DAG: v_mov_b32_e32 v0, v2
; GCN-DAG: v_mov_b32_e32 v1, v3
; GCN: v_mqsad_u32_u8 v[2:5], v[0:1], v6, v[{{[0-9]+:[0-9]+}}]
define amdgpu_kernel void @v_mqsad_u32_u8_non_immediate(<4 x i32> addrspace(1)* %out, i64 %src, i32 %a, <4 x i32> %b) {
  %tmp = call i64 asm "v_lsrlrev_b64 $0, $1, 1", "={v[2:3]},v"(i64 %src) #0
  %tmp1 = call i32 asm "v_mov_b32 $0, $1", "={v4},v"(i32 %a) #0
  %tmp2 = call <4 x i32> @llvm.amdgcn.mqsad.u32.u8(i64 %tmp, i32 %tmp1, <4 x i32> %b) #0
  %tmp3 = call <4 x i32>  asm ";; force constraint", "=v,{v[2:5]}"(<4 x i32> %tmp2) #0
  store <4 x i32> %tmp3, <4 x i32> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_mqsad_u32_u8_inline_fp_immediate:
; GCN-DAG: v_mov_b32_e32 v0, v2
; GCN-DAG: v_mov_b32_e32 v1, v3
; GCN: v_mqsad_u32_u8 v[2:5], v[0:1], v6, v[{{[0-9]+:[0-9]+}}]
define amdgpu_kernel void @v_mqsad_u32_u8_inline_fp_immediate(<4 x i32> addrspace(1)* %out, i64 %src, i32 %a) {
  %tmp = call i64 asm "v_lsrlrev_b64 $0, $1, 1", "={v[2:3]},v"(i64 %src) #0
  %tmp1 = call i32 asm "v_mov_b32 $0, $1", "={v4},v"(i32 %a) #0
  %tmp2 = call <4 x i32> @llvm.amdgcn.mqsad.u32.u8(i64 %tmp, i32 %tmp1, <4 x i32> <i32 1065353216, i32 0, i32 0, i32 0>) #0
  %tmp3 = call <4 x i32>  asm ";; force constraint", "=v,{v[2:5]}"(<4 x i32> %tmp2) #0
  store <4 x i32> %tmp3, <4 x i32> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_mqsad_u32_u8_use_sgpr_vgpr:
; GCN-DAG: v_mov_b32_e32 v0, v2
; GCN-DAG: v_mov_b32_e32 v1, v3
; GCN: v_mqsad_u32_u8 v[2:5], v[0:1], v6, v[{{[0-9]+:[0-9]+}}]
define amdgpu_kernel void @v_mqsad_u32_u8_use_sgpr_vgpr(<4 x i32> addrspace(1)* %out, i64 %src, i32 %a, <4 x i32> addrspace(1)* %input) {
  %in = load <4 x i32>, <4 x i32> addrspace(1) * %input
  %tmp = call i64 asm "v_lsrlrev_b64 $0, $1, 1", "={v[2:3]},v"(i64 %src) #0
  %tmp1 = call i32 asm "v_mov_b32 $0, $1", "={v4},v"(i32 %a) #0
  %tmp2 = call <4 x i32> @llvm.amdgcn.mqsad.u32.u8(i64 %tmp, i32 %tmp1, <4 x i32> %in) #0
  %tmp3 = call <4 x i32>  asm ";; force constraint", "=v,{v[2:5]}"(<4 x i32> %tmp2) #0
  store <4 x i32> %tmp3, <4 x i32> addrspace(1)* %out, align 4
  ret void
}

attributes #0 = { nounwind readnone }
