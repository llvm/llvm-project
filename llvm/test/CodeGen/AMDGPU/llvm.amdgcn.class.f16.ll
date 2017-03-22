; RUN: llc -march=amdgcn -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

declare half @llvm.fabs.f16(half %a)
declare i1 @llvm.amdgcn.class.f16(half %a, i32 %b)

; GCN-LABEL: {{^}}class_f16
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_dword v[[B_I32:[0-9]+]]
; VI:  v_cmp_class_f16_e32 vcc, v[[A_F16]], v[[B_I32]]
; GCN: v_cndmask_b32_e64 v[[R_I32:[0-9]+]]
; GCN: buffer_store_dword v[[R_I32]]
; GCN: s_endpgm
define amdgpu_kernel void @class_f16(
    i32 addrspace(1)* %r,
    half addrspace(1)* %a,
    i32 addrspace(1)* %b) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load i32, i32 addrspace(1)* %b
  %r.val = call i1 @llvm.amdgcn.class.f16(half %a.val, i32 %b.val)
  %r.val.sext = sext i1 %r.val to i32
  store i32 %r.val.sext, i32 addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}class_f16_fabs
; GCN: s_load_dword s[[SA_F16:[0-9]+]]
; GCN: s_load_dword s[[SB_I32:[0-9]+]]
; VI:  v_trunc_f16_e32 v[[VA_F16:[0-9]+]], s[[SA_F16]]
; VI:  v_cmp_class_f16_e64 [[CMP:s\[[0-9]+:[0-9]+\]]], |v[[VA_F16]]|, s[[SB_I32]]
; VI:  v_cndmask_b32_e64 v[[VR_I32:[0-9]+]], 0, -1, [[CMP]]
; GCN: buffer_store_dword v[[VR_I32]]
; GCN: s_endpgm
define amdgpu_kernel void @class_f16_fabs(
  i32 addrspace(1)* %r,
  half %a.val,
  i32 %b.val) {
entry:
  %a.val.fabs = call half @llvm.fabs.f16(half %a.val)
  %r.val = call i1 @llvm.amdgcn.class.f16(half %a.val.fabs, i32 %b.val)
  %r.val.sext = sext i1 %r.val to i32
  store i32 %r.val.sext, i32 addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}class_f16_fneg
; GCN: s_load_dword s[[SA_F16:[0-9]+]]
; GCN: s_load_dword s[[SB_I32:[0-9]+]]
; VI:  v_trunc_f16_e64 v[[VA_F16:[0-9]+]], -s[[SA_F16]]
; VI:  v_cmp_class_f16_e64 [[CMP:s\[[0-9]+:[0-9]+\]]], v[[VA_F16]], s[[SB_I32]]
; VI:  v_cndmask_b32_e64 v[[VR_I32:[0-9]+]], 0, -1, [[CMP]]
; GCN: buffer_store_dword v[[VR_I32]]
; GCN: s_endpgm
define amdgpu_kernel void @class_f16_fneg(
  i32 addrspace(1)* %r,
  half %a.val,
  i32 %b.val) {
entry:
  %a.val.fneg = fsub half -0.0, %a.val
  %r.val = call i1 @llvm.amdgcn.class.f16(half %a.val.fneg, i32 %b.val)
  %r.val.sext = sext i1 %r.val to i32
  store i32 %r.val.sext, i32 addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}class_f16_fabs_fneg
; GCN: s_load_dword s[[SA_F16:[0-9]+]]
; GCN: s_load_dword s[[SB_I32:[0-9]+]]
; VI:  v_trunc_f16_e32 v[[VA_F16:[0-9]+]], s[[SA_F16]]
; VI:  v_cmp_class_f16_e64 [[CMP:s\[[0-9]+:[0-9]+\]]], -|v[[VA_F16]]|, s[[SB_I32]]
; VI:  v_cndmask_b32_e64 v[[VR_I32:[0-9]+]], 0, -1, [[CMP]]
; GCN: buffer_store_dword v[[VR_I32]]
; GCN: s_endpgm
define amdgpu_kernel void @class_f16_fabs_fneg(
  i32 addrspace(1)* %r,
  half %a.val,
  i32 %b.val) {
entry:
  %a.val.fabs = call half @llvm.fabs.f16(half %a.val)
  %a.val.fabs.fneg = fsub half -0.0, %a.val.fabs
  %r.val = call i1 @llvm.amdgcn.class.f16(half %a.val.fabs.fneg, i32 %b.val)
  %r.val.sext = sext i1 %r.val to i32
  store i32 %r.val.sext, i32 addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}class_f16_1
; GCN: s_load_dword s[[SA_F16:[0-9]+]]
; VI:  v_trunc_f16_e32 v[[VA_F16:[0-9]+]], s[[SA_F16]]
; VI:  v_cmp_class_f16_e64 [[CMP:s\[[0-9]+:[0-9]+\]]], v[[VA_F16]], 1{{$}}
; VI:  v_cndmask_b32_e64 v[[VR_I32:[0-9]+]], 0, -1, [[CMP]]
; GCN: buffer_store_dword v[[VR_I32]]
; GCN: s_endpgm
define amdgpu_kernel void @class_f16_1(
  i32 addrspace(1)* %r,
  half %a.val) {
entry:
  %r.val = call i1 @llvm.amdgcn.class.f16(half %a.val, i32 1)
  %r.val.sext = sext i1 %r.val to i32
  store i32 %r.val.sext, i32 addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}class_f16_64
; GCN: s_load_dword s[[SA_F16:[0-9]+]]
; VI:  v_trunc_f16_e32 v[[VA_F16:[0-9]+]], s[[SA_F16]]
; VI:  v_cmp_class_f16_e64 [[CMP:s\[[0-9]+:[0-9]+\]]], v[[VA_F16]], 64{{$}}
; VI:  v_cndmask_b32_e64 v[[VR_I32:[0-9]+]], 0, -1, [[CMP]]
; GCN: buffer_store_dword v[[VR_I32]]
; GCN: s_endpgm
define amdgpu_kernel void @class_f16_64(
  i32 addrspace(1)* %r,
  half %a.val) {
entry:
  %r.val = call i1 @llvm.amdgcn.class.f16(half %a.val, i32 64)
  %r.val.sext = sext i1 %r.val to i32
  store i32 %r.val.sext, i32 addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}class_f16_full_mask
; GCN: s_load_dword s[[SA_F16:[0-9]+]]
; VI:  v_mov_b32_e32 v[[MASK:[0-9]+]], 0x3ff{{$}}
; VI:  v_trunc_f16_e32 v[[VA_F16:[0-9]+]], s[[SA_F16]]
; VI:  v_cmp_class_f16_e32 vcc, v[[VA_F16]], v[[MASK]]
; VI:  v_cndmask_b32_e64 v[[VR_I32:[0-9]+]], 0, -1, vcc
; GCN: buffer_store_dword v[[VR_I32]]
; GCN: s_endpgm
define amdgpu_kernel void @class_f16_full_mask(
  i32 addrspace(1)* %r,
  half %a.val) {
entry:
  %r.val = call i1 @llvm.amdgcn.class.f16(half %a.val, i32 1023)
  %r.val.sext = sext i1 %r.val to i32
  store i32 %r.val.sext, i32 addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}class_f16_nine_bit_mask
; GCN: s_load_dword s[[SA_F16:[0-9]+]]
; VI:  v_mov_b32_e32 v[[MASK:[0-9]+]], 0x1ff{{$}}
; VI:  v_trunc_f16_e32 v[[VA_F16:[0-9]+]], s[[SA_F16]]
; VI:  v_cmp_class_f16_e32 vcc, v[[VA_F16]], v[[MASK]]
; VI:  v_cndmask_b32_e64 v[[VR_I32:[0-9]+]], 0, -1, vcc
; GCN: buffer_store_dword v[[VR_I32]]
; GCN: s_endpgm
define amdgpu_kernel void @class_f16_nine_bit_mask(
  i32 addrspace(1)* %r,
  half %a.val) {
entry:
  %r.val = call i1 @llvm.amdgcn.class.f16(half %a.val, i32 511)
  %r.val.sext = sext i1 %r.val to i32
  store i32 %r.val.sext, i32 addrspace(1)* %r
  ret void
}
