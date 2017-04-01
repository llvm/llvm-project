; RUN: llc -march=amdgcn -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -check-prefix=GCN -check-prefix=SIVI %s
; RUN: llc -march=amdgcn -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -check-prefix=GCN -check-prefix=SIVI %s
; RUN: llc -march=amdgcn -mcpu=gfx901 -mattr=-flat-for-global,+fp64-fp16-denormals -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -check-prefix=GCN -check-prefix=GFX9 -check-prefix=GFX9-DENORM %s
; RUN: llc -march=amdgcn -mcpu=gfx901 -mattr=-flat-for-global,-fp64-fp16-denormals -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -check-prefix=GCN -check-prefix=GFX9 -check-prefix=GFX9-FLUSH %s

; GCN-LABEL: {{^}}fptrunc_f32_to_f16:
; GCN: buffer_load_dword v[[A_F32:[0-9]+]]
; GCN: v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[A_F32]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @fptrunc_f32_to_f16(
    half addrspace(1)* %r,
    float addrspace(1)* %a) {
entry:
  %a.val = load float, float addrspace(1)* %a
  %r.val = fptrunc float %a.val to half
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fptrunc_f64_to_f16:
; GCN: buffer_load_dwordx2 v{{\[}}[[A_F64_0:[0-9]+]]:[[A_F64_1:[0-9]+]]{{\]}}
; GCN: v_cvt_f32_f64_e32 v[[A_F32:[0-9]+]], v{{\[}}[[A_F64_0]]:[[A_F64_1]]{{\]}}
; GCN: v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[A_F32]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @fptrunc_f64_to_f16(
    half addrspace(1)* %r,
    double addrspace(1)* %a) {
entry:
  %a.val = load double, double addrspace(1)* %a
  %r.val = fptrunc double %a.val to half
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fptrunc_v2f32_to_v2f16:
; GCN:     buffer_load_dwordx2 v{{\[}}[[A_F32_0:[0-9]+]]:[[A_F32_1:[0-9]+]]{{\]}}
; GCN-DAG: v_cvt_f16_f32_e32 v[[R_F16_0:[0-9]+]], v[[A_F32_0]]
; GCN-DAG: v_cvt_f16_f32_e32 v[[R_F16_1:[0-9]+]], v[[A_F32_1]]
; SIVI-DAG: v_lshlrev_b32_e32 v[[R_F16_HI:[0-9]+]], 16, v[[R_F16_1]]
; SIVI:     v_or_b32_e32 v[[R_V2_F16:[0-9]+]], v[[R_F16_HI]], v[[R_F16_0]]

; GFX9-FLUSH: v_and_b32_e32 v[[R_F16_LO:[0-9]+]], 0xffff, v[[R_F16_0]]
; GFX9-FLUSH: v_lshl_or_b32 v[[R_V2_F16:[0-9]+]], v[[R_F16_1]], 16, v[[R_F16_LO]]

; GFX9-DENORM: v_pack_b32_f16 v[[R_V2_F16:[0-9]+]], v[[R_F16_0]], v[[R_F16_1]]

; GCN:     buffer_store_dword v[[R_V2_F16]]
; GCN:     s_endpgm
define amdgpu_kernel void @fptrunc_v2f32_to_v2f16(
    <2 x half> addrspace(1)* %r,
    <2 x float> addrspace(1)* %a) {
entry:
  %a.val = load <2 x float>, <2 x float> addrspace(1)* %a
  %r.val = fptrunc <2 x float> %a.val to <2 x half>
  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fptrunc_v2f64_to_v2f16:
; GCN: buffer_load_dwordx4 v{{\[}}[[A_F64_0:[0-9]+]]:[[A_F64_3:[0-9]+]]{{\]}}
; GCN-DAG: v_cvt_f32_f64_e32 v[[A_F32_0:[0-9]+]], v{{\[}}[[A_F64_0]]:{{[0-9]+}}{{\]}}
; GCN-DAG: v_cvt_f32_f64_e32 v[[A_F32_1:[0-9]+]], v{{\[}}{{[0-9]+}}:[[A_F64_3]]{{\]}}
; GCN-DAG: v_cvt_f16_f32_e32 v[[R_F16_0:[0-9]+]], v[[A_F32_0]]
; GCN-DAG: v_cvt_f16_f32_e32 v[[R_F16_1:[0-9]+]], v[[A_F32_1]]

; SIVI: v_lshlrev_b32_e32 v[[R_F16_HI:[0-9]+]], 16, v[[R_F16_1]]
; SIVI: v_or_b32_e32 v[[R_V2_F16:[0-9]+]], v[[R_F16_HI]], v[[R_F16_0]]

; GFX9-FLUSH: v_and_b32_e32 v[[R_F16_LO:[0-9]+]], 0xffff, v[[R_F16_0]]
; GFX9-FLUSH: v_lshl_or_b32 v[[R_V2_F16:[0-9]+]], v[[R_F16_1]], 16, v[[R_F16_LO]]

; GFX9-DENORM: v_pack_b32_f16 v[[R_V2_F16:[0-9]+]], v[[R_F16_0]], v[[R_F16_1]]

; GCN: buffer_store_dword v[[R_V2_F16]]
define amdgpu_kernel void @fptrunc_v2f64_to_v2f16(
    <2 x half> addrspace(1)* %r,
    <2 x double> addrspace(1)* %a) {
entry:
  %a.val = load <2 x double>, <2 x double> addrspace(1)* %a
  %r.val = fptrunc <2 x double> %a.val to <2 x half>
  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fneg_fptrunc_f32_to_f16:
; GCN: buffer_load_dword v[[A_F32:[0-9]+]]
; GCN: v_cvt_f16_f32_e64 v[[R_F16:[0-9]+]], -v[[A_F32]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @fneg_fptrunc_f32_to_f16(
    half addrspace(1)* %r,
    float addrspace(1)* %a) {
entry:
  %a.val = load float, float addrspace(1)* %a
  %a.fneg = fsub float -0.0, %a.val
  %r.val = fptrunc float %a.fneg to half
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fabs_fptrunc_f32_to_f16:
; GCN: buffer_load_dword v[[A_F32:[0-9]+]]
; GCN: v_cvt_f16_f32_e64 v[[R_F16:[0-9]+]], |v[[A_F32]]|
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @fabs_fptrunc_f32_to_f16(
    half addrspace(1)* %r,
    float addrspace(1)* %a) {
entry:
  %a.val = load float, float addrspace(1)* %a
  %a.fabs = call float @llvm.fabs.f32(float %a.val)
  %r.val = fptrunc float %a.fabs to half
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fneg_fabs_fptrunc_f32_to_f16:
; GCN: buffer_load_dword v[[A_F32:[0-9]+]]
; GCN: v_cvt_f16_f32_e64 v[[R_F16:[0-9]+]], -|v[[A_F32]]|
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @fneg_fabs_fptrunc_f32_to_f16(
    half addrspace(1)* %r,
    float addrspace(1)* %a) #0 {
entry:
  %a.val = load float, float addrspace(1)* %a
  %a.fabs = call float @llvm.fabs.f32(float %a.val)
  %a.fneg.fabs = fsub float -0.0, %a.fabs
  %r.val = fptrunc float %a.fneg.fabs to half
  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fptrunc_f32_to_f16_zext_i32:
; GCN: buffer_load_dword v[[A_F32:[0-9]+]]
; GCN: v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[A_F32]]
; GCN-NOT: v[[R_F16]]
; GCN: buffer_store_dword v[[R_F16]]
define amdgpu_kernel void @fptrunc_f32_to_f16_zext_i32(
    i32 addrspace(1)* %r,
    float addrspace(1)* %a) #0 {
entry:
  %a.val = load float, float addrspace(1)* %a
  %r.val = fptrunc float %a.val to half
  %r.i16 = bitcast half %r.val to i16
  %zext = zext i16 %r.i16 to i32
  store i32 %zext, i32 addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fptrunc_fabs_f32_to_f16_zext_i32:
; GCN: buffer_load_dword v[[A_F32:[0-9]+]]
; GCN: v_cvt_f16_f32_e64 v[[R_F16:[0-9]+]], |v[[A_F32]]|
; GCN-NOT: v[[R_F16]]
; GCN: buffer_store_dword v[[R_F16]]
define amdgpu_kernel void @fptrunc_fabs_f32_to_f16_zext_i32(
    i32 addrspace(1)* %r,
    float addrspace(1)* %a) #0 {
entry:
  %a.val = load float, float addrspace(1)* %a
  %a.fabs = call float @llvm.fabs.f32(float %a.val)
  %r.val = fptrunc float %a.fabs to half
  %r.i16 = bitcast half %r.val to i16
  %zext = zext i16 %r.i16 to i32
  store i32 %zext, i32 addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}fptrunc_f32_to_f16_sext_i32:
; GCN: buffer_load_dword v[[A_F32:[0-9]+]]
; GCN: v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[A_F32]]
; GCN: v_bfe_i32 v[[R_F16_SEXT:[0-9]+]], v[[R_F16]], 0, 16
; GCN: buffer_store_dword v[[R_F16_SEXT]]
define amdgpu_kernel void @fptrunc_f32_to_f16_sext_i32(
    i32 addrspace(1)* %r,
    float addrspace(1)* %a) #0 {
entry:
  %a.val = load float, float addrspace(1)* %a
  %r.val = fptrunc float %a.val to half
  %r.i16 = bitcast half %r.val to i16
  %zext = sext i16 %r.i16 to i32
  store i32 %zext, i32 addrspace(1)* %r
  ret void
}

declare float @llvm.fabs.f32(float) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
