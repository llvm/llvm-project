; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare i64 @llvm.amdgcn.mqsad.pk.u16.u8(i64, i32, i64) #0

; GCN-LABEL: {{^}}v_mqsad_pk_u16_u8:
; GCN: v_mqsad_pk_u16_u8 v[0:1], v[4:5], s{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
; GCN-DAG: v_mov_b32_e32 v5, v1
; GCN-DAG: v_mov_b32_e32 v4, v0
define amdgpu_kernel void @v_mqsad_pk_u16_u8(ptr addrspace(1) %out, i64 %src) {
  %tmp = call i64 asm "v_lsrlrev_b64 $0, $1, 1", "={v[4:5]},v"(i64 %src) #0
  %tmp1 = call i64 @llvm.amdgcn.mqsad.pk.u16.u8(i64 %tmp, i32 100, i64 100) #0
  %tmp2 = call i64 asm ";; force constraint", "=v,{v[4:5]}"(i64 %tmp1) #0
  store i64 %tmp2, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_mqsad_pk_u16_u8_non_immediate:
; GCN: v_mqsad_pk_u16_u8 v[0:1], v[2:3], v4, v[6:7]
; GCN-DAG: v_mov_b32_e32 v3, v1
; GCN-DAG: v_mov_b32_e32 v2, v0
define amdgpu_kernel void @v_mqsad_pk_u16_u8_non_immediate(ptr addrspace(1) %out, i64 %src, i32 %a, i64 %b) {
  %tmp = call i64 asm "v_lsrlrev_b64 $0, $1, 1", "={v[2:3]},v"(i64 %src) #0
  %tmp1 = call i32 asm "v_mov_b32 $0, $1", "={v4},v"(i32 %a) #0
  %tmp2 = call i64 asm "v_lshlrev_b64 $0, $1, 1", "={v[6:7]},v"(i64 %b) #0
  %tmp3 = call i64 @llvm.amdgcn.mqsad.pk.u16.u8(i64 %tmp, i32 %tmp1, i64 %tmp2) #0
  %tmp4 = call i64 asm ";; force constraint", "=v,{v[2:3]}"(i64 %tmp3) #0
  store i64 %tmp4, ptr addrspace(1) %out, align 4
  ret void
}

attributes #0 = { nounwind readnone }
