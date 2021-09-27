; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tahiti -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -allow-deprecated-dag-overlap -enable-var-scope --check-prefixes=GCN,SI %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -allow-deprecated-dag-overlap -enable-var-scope --check-prefixes=GCN,GFX89 %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -allow-deprecated-dag-overlap -enable-var-scope --check-prefixes=GCN,GFX89 %s

; GCN-LABEL: {{^}}fpext_f16_to_f32
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: v_cvt_f32_f16_e32 v[[R_F32:[0-9]+]], v[[A_F16]]
; GCN: buffer_store_dword v[[R_F32]]
; GCN: s_endpgm
define amdgpu_kernel void @fpext_f16_to_f32(
    float addrspace(1)* %r,
    half addrspace(1)* %a) #0 {
entry:
  %a.val = load half, half addrspace(1)* %a
  %r.val = fpext half %a.val to float
  store float %r.val, float addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fpext_f16_to_f64
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; GCN: v_cvt_f64_f32_e32 v{{\[}}[[R_F64_0:[0-9]+]]:[[R_F64_1:[0-9]+]]{{\]}}, v[[A_F32]]
; GCN: buffer_store_dwordx2 v{{\[}}[[R_F64_0]]:[[R_F64_1]]{{\]}}
; GCN: s_endpgm
define amdgpu_kernel void @fpext_f16_to_f64(
    double addrspace(1)* %r,
    half addrspace(1)* %a) #0 {
entry:
  %a.val = load half, half addrspace(1)* %a
  %r.val = fpext half %a.val to double
  store double %r.val, double addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fpext_v2f16_to_v2f32
; GCN: buffer_load_dword v[[A_V2_F16:[0-9]+]]
; GCN-DAG: v_cvt_f32_f16_e32 v[[R_F32_0:[0-9]+]], v[[A_V2_F16]]
; SI:  v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; SI: v_cvt_f32_f16_e32 v[[R_F32_1:[0-9]+]], v[[A_F16_1]]
; GFX89: v_cvt_f32_f16_sdwa v[[R_F32_1:[0-9]+]], v[[A_V2_F16]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
; GCN: buffer_store_dwordx2 v{{\[}}[[R_F32_0]]:[[R_F32_1]]{{\]}}
; GCN: s_endpgm

define amdgpu_kernel void @fpext_v2f16_to_v2f32(
    <2 x float> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a) #0 {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %r.val = fpext <2 x half> %a.val to <2 x float>
  store <2 x float> %r.val, <2 x float> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fpext_v2f16_to_v2f64
; GCN: buffer_load_dword
; SI-DAG: v_lshrrev_b32_e32
; SI-DAG: v_cvt_f32_f16_e32
; GCN: v_cvt_f32_f16_e32
; GFX89: v_cvt_f32_f16_sdwa

; GCN: v_cvt_f64_f32_e32
; GCN: v_cvt_f64_f32_e32
; GCN: buffer_store_dwordx4
; GCN: s_endpgm

define amdgpu_kernel void @fpext_v2f16_to_v2f64(
    <2 x double> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a) {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %r.val = fpext <2 x half> %a.val to <2 x double>
  store <2 x double> %r.val, <2 x double> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}s_fneg_fpext_f16_to_f32:
; GCN: v_cvt_f32_f16_e32 v{{[0-9]+}}, s{{[0-9]+}}
define amdgpu_kernel void @s_fneg_fpext_f16_to_f32(float addrspace(1)* %r, i32 %a) {
entry:
  %a.trunc = trunc i32 %a to i16
  %a.val = bitcast i16 %a.trunc to half
  %r.val = fpext half %a.val to float
  store float %r.val, float addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fneg_fpext_f16_to_f32:
; GCN: {{buffer|flat}}_load_ushort [[A:v[0-9]+]]
; GCN: v_cvt_f32_f16_e64 v{{[0-9]+}}, -[[A]]
define amdgpu_kernel void @fneg_fpext_f16_to_f32(
    float addrspace(1)* %r,
    half addrspace(1)* %a) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %a.neg = fsub half -0.0, %a.val
  %r.val = fpext half %a.neg to float
  store float %r.val, float addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fabs_fpext_f16_to_f32:
; GCN: {{buffer|flat}}_load_ushort [[A:v[0-9]+]]
; GCN: v_cvt_f32_f16_e64 v{{[0-9]+}}, |[[A]]|
define amdgpu_kernel void @fabs_fpext_f16_to_f32(
    float addrspace(1)* %r,
    half addrspace(1)* %a) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %a.fabs = call half @llvm.fabs.f16(half %a.val)
  %r.val = fpext half %a.fabs to float
  store float %r.val, float addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fneg_fabs_fpext_f16_to_f32:
; GCN: {{buffer|flat}}_load_ushort [[A:v[0-9]+]]
; GCN: v_cvt_f32_f16_e64 v{{[0-9]+}}, -|[[A]]|
define amdgpu_kernel void @fneg_fabs_fpext_f16_to_f32(
    float addrspace(1)* %r,
    half addrspace(1)* %a) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %a.fabs = call half @llvm.fabs.f16(half %a.val)
  %a.fneg.fabs = fsub half -0.0, %a.fabs
  %r.val = fpext half %a.fneg.fabs to float
  store float %r.val, float addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fneg_multi_use_fpext_f16_to_f32:
; GCN: {{buffer|flat}}_load_ushort [[A:v[0-9]+]]
; GCN-DAG: v_xor_b32_e32 [[XOR:v[0-9]+]], 0x8000, [[A]]

; FIXME: Using the source modifier here only wastes code size
; SI-DAG: v_cvt_f32_f16_e32 [[CVT:v[0-9]+]], [[A]]
; GFX89-DAG: v_cvt_f32_f16_e64 [[CVT:v[0-9]+]], -[[A]]

; GCN: store_dword [[CVT]]
; GCN: store_short [[XOR]]
define amdgpu_kernel void @fneg_multi_use_fpext_f16_to_f32(
    float addrspace(1)* %r,
    half addrspace(1)* %a) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %a.neg = fsub half -0.0, %a.val
  %r.val = fpext half %a.neg to float
  store volatile float %r.val, float addrspace(1)* %r
  store volatile half %a.neg, half addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}fneg_multi_foldable_use_fpext_f16_to_f32:
; GCN: {{buffer|flat}}_load_ushort [[A:v[0-9]+]]
; GCN-DAG: v_cvt_f32_f16_e64 [[CVTA_NEG:v[0-9]+]], -[[A]]
; SI-DAG: v_cvt_f32_f16_e32 [[CVTA:v[0-9]+]], [[A]]
; SI: v_mul_f32_e32 [[MUL_F32:v[0-9]+]], [[CVTA_NEG]], [[CVTA]]
; SI: v_cvt_f16_f32_e32 [[MUL:v[0-9]+]], [[MUL_F32]]

; GFX89-DAG: v_cvt_f32_f16_e64 [[CVT_NEGA:v[0-9]+]], -[[A]]
; GFX89: v_mul_f16_e64 [[MUL:v[0-9]+]], -[[A]], [[A]]

; GCN: buffer_store_dword [[CVTA_NEG]]
; GCN: buffer_store_short [[MUL]]
define amdgpu_kernel void @fneg_multi_foldable_use_fpext_f16_to_f32(
    float addrspace(1)* %r,
    half addrspace(1)* %a) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %a.neg = fsub half -0.0, %a.val
  %r.val = fpext half %a.neg to float
  %mul = fmul half %a.neg, %a.val
  store volatile float %r.val, float addrspace(1)* %r
  store volatile half %mul, half addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}fabs_multi_use_fpext_f16_to_f32:
; GCN: {{buffer|flat}}_load_ushort [[A:v[0-9]+]]
; GCN-DAG: v_and_b32_e32 [[XOR:v[0-9]+]], 0x7fff, [[A]]

; SI-DAG: v_cvt_f32_f16_e32 [[CVT:v[0-9]+]], [[A]]
; GFX89-DAG: v_cvt_f32_f16_e64 [[CVT:v[0-9]+]], |[[A]]|

; GCN: store_dword [[CVT]]
; GCN: store_short [[XOR]]
define amdgpu_kernel void @fabs_multi_use_fpext_f16_to_f32(
    float addrspace(1)* %r,
    half addrspace(1)* %a) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %a.fabs = call half @llvm.fabs.f16(half %a.val)
  %r.val = fpext half %a.fabs to float
  store volatile float %r.val, float addrspace(1)* %r
  store volatile half %a.fabs, half addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}fabs_multi_foldable_use_fpext_f16_to_f32:
; GCN: {{buffer|flat}}_load_ushort [[A:v[0-9]+]]
; SI: v_cvt_f32_f16_e32 [[CVTA:v[0-9]+]], [[A]]
; SI: v_mul_f32_e64 [[MUL_F32:v[0-9]+]], |[[CVTA]]|, [[CVTA]]
; SI: v_cvt_f16_f32_e32 [[MUL:v[0-9]+]], [[MUL_F32]]
; SI: v_and_b32_e32 [[ABS_A:v[0-9]+]], 0x7fffffff, [[CVTA]]

; GFX89-DAG: v_cvt_f32_f16_e64 [[ABS_A:v[0-9]+]], |[[A]]|
; GFX89: v_mul_f16_e64 [[MUL:v[0-9]+]], |[[A]]|, [[A]]

; GCN: buffer_store_dword [[ABS_A]]
; GCN: buffer_store_short [[MUL]]
define amdgpu_kernel void @fabs_multi_foldable_use_fpext_f16_to_f32(
    float addrspace(1)* %r,
    half addrspace(1)* %a) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %a.fabs = call half @llvm.fabs.f16(half %a.val)
  %r.val = fpext half %a.fabs to float
  %mul = fmul half %a.fabs, %a.val
  store volatile float %r.val, float addrspace(1)* %r
  store volatile half %mul, half addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}fabs_fneg_multi_use_fpext_f16_to_f32:
; GCN: {{buffer|flat}}_load_ushort [[A:v[0-9]+]]
; GCN-DAG: v_or_b32_e32 [[OR:v[0-9]+]], 0x8000, [[A]]

; SI: v_cvt_f32_f16_e32 [[CVT:v[0-9]+]], [[OR]]
; GFX89-DAG: v_cvt_f32_f16_e64 [[CVT:v[0-9]+]], -|[[OR]]|

; GCN: buffer_store_dword [[CVT]]
; GCN: buffer_store_short [[OR]]
define amdgpu_kernel void @fabs_fneg_multi_use_fpext_f16_to_f32(
    float addrspace(1)* %r,
    half addrspace(1)* %a) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %a.fabs = call half @llvm.fabs.f16(half %a.val)
  %a.fneg.fabs = fsub half -0.0, %a.fabs
  %r.val = fpext half %a.fneg.fabs to float
  store volatile float %r.val, float addrspace(1)* %r
  store volatile half %a.fneg.fabs, half addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}fabs_fneg_multi_foldable_use_fpext_f16_to_f32:
; GCN: {{buffer|flat}}_load_ushort [[A:v[0-9]+]]
; SI: v_cvt_f32_f16_e32 [[CVTA:v[0-9]+]], [[A]]
; SI: v_mul_f32_e64 [[MUL_F32:v[0-9]+]], -|[[CVTA]]|, [[CVTA]]
; SI: v_cvt_f16_f32_e32 [[MUL:v[0-9]+]], [[MUL_F32]]
; SI: v_or_b32_e32 [[FABS_FNEG:v[0-9]+]], 0x80000000, [[CVTA]]

; GFX89-DAG: v_cvt_f32_f16_e64 [[FABS_FNEG:v[0-9]+]], -|[[A]]|
; GFX89-DAG: v_mul_f16_e64 [[MUL:v[0-9]+]], -|[[A]]|, [[A]]

; GCN: buffer_store_dword [[FABS_FNEG]]
; GCN: buffer_store_short [[MUL]]
define amdgpu_kernel void @fabs_fneg_multi_foldable_use_fpext_f16_to_f32(
    float addrspace(1)* %r,
    half addrspace(1)* %a) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %a.fabs = call half @llvm.fabs.f16(half %a.val)
  %a.fneg.fabs = fsub half -0.0, %a.fabs
  %r.val = fpext half %a.fneg.fabs to float
  %mul = fmul half %a.fneg.fabs, %a.val
  store volatile float %r.val, float addrspace(1)* %r
  store volatile half %mul, half addrspace(1)* undef
  ret void
}

declare half @llvm.fabs.f16(half) #1

attributes #1 = { nounwind readnone }
