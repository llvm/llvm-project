; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,SI %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,GFX89 %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,GFX89 %s

declare half @llvm.copysign.f16(half, half)
declare float @llvm.copysign.f32(float, float)
declare double @llvm.copysign.f64(double, double)
declare <2 x half> @llvm.copysign.v2f16(<2 x half>, <2 x half>)
declare <3 x half> @llvm.copysign.v3f16(<3 x half>, <3 x half>)
declare <4 x half> @llvm.copysign.v4f16(<4 x half>, <4 x half>)

declare i32 @llvm.amdgcn.workitem.id.x()

; GCN-LABEL: {{^}}test_copysign_f16:
; SI: {{buffer|flat}}_load_ushort v[[MAG:[0-9]+]]
; SI: {{buffer|flat}}_load_ushort v[[SIGN:[0-9]+]]
; SI: s_brev_b32 s[[CONST:[0-9]+]], -2
; SI-DAG: v_cvt_f32_f16_e32 v[[MAG_F32:[0-9]+]], v[[MAG]]
; SI-DAG: v_cvt_f32_f16_e32 v[[SIGN_F32:[0-9]+]], v[[SIGN]]
; SI: v_bfi_b32 v[[OUT_F32:[0-9]+]], s[[CONST]], v[[MAG_F32]], v[[SIGN_F32]]
; SI: v_cvt_f16_f32_e32 v[[OUT:[0-9]+]], v[[OUT_F32]]
; GFX89: {{buffer|flat}}_load_ushort v[[MAG:[0-9]+]]
; GFX89: {{buffer|flat}}_load_ushort v[[SIGN:[0-9]+]]
; GFX89: s_movk_i32 s[[CONST:[0-9]+]], 0x7fff
; GFX89: v_bfi_b32 v[[OUT:[0-9]+]], s[[CONST]], v[[MAG]], v[[SIGN]]
; GCN: buffer_store_short v[[OUT]]
; GCN: s_endpgm
define amdgpu_kernel void @test_copysign_f16(
  half addrspace(1)* %arg_out,
  half addrspace(1)* %arg_mag,
  half addrspace(1)* %arg_sign) {
entry:
  %mag = load volatile half, half addrspace(1)* %arg_mag
  %sign = load volatile half, half addrspace(1)* %arg_sign
  %out = call half @llvm.copysign.f16(half %mag, half %sign)
  store half %out, half addrspace(1)* %arg_out
  ret void
}

; GCN-LABEL: {{^}}test_copysign_out_f32_mag_f16_sign_f32:
; GCN-DAG: {{buffer|flat|global}}_load_ushort v[[MAG:[0-9]+]]
; GCN-DAG: {{buffer|flat|global}}_load_dword v[[SIGN:[0-9]+]]
; GCN-DAG: s_brev_b32 s[[CONST:[0-9]+]], -2
; GCN-DAG: v_cvt_f32_f16_e32 v[[MAG_EXT:[0-9]+]], v[[MAG]]
; GCN: v_bfi_b32 v[[OUT:[0-9]+]], s[[CONST]], v[[MAG_EXT]], v[[SIGN]]
; GCN: buffer_store_dword v[[OUT]]
; GCN: s_endpgm
define amdgpu_kernel void @test_copysign_out_f32_mag_f16_sign_f32(
  float addrspace(1)* %arg_out,
  half addrspace(1)* %arg_mag,
  float addrspace(1)* %arg_sign) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %arg_mag_gep = getelementptr half, half addrspace(1)* %arg_mag, i32 %tid
  %mag = load half, half addrspace(1)* %arg_mag_gep
  %mag.ext = fpext half %mag to float
  %arg_sign_gep = getelementptr float, float addrspace(1)* %arg_sign, i32 %tid
  %sign = load float, float addrspace(1)* %arg_sign_gep
  %out = call float @llvm.copysign.f32(float %mag.ext, float %sign)
  store float %out, float addrspace(1)* %arg_out
  ret void
}

; GCN-LABEL: {{^}}test_copysign_out_f64_mag_f16_sign_f64:
; GCN-DAG: {{buffer|flat|global}}_load_ushort v[[MAG:[0-9]+]]
; GCN-DAG: {{buffer|flat|global}}_load_dwordx2 v{{\[}}[[SIGN_LO:[0-9]+]]:[[SIGN_HI:[0-9]+]]{{\]}}
; GCN-DAG: s_brev_b32 s[[CONST:[0-9]+]], -2
; GCN-DAG: v_cvt_f32_f16_e32 v[[MAG_EXT:[0-9]+]], v[[MAG]]
; GCN-DAG: v_cvt_f64_f32_e32 v{{\[}}[[MAG_EXT_LO:[0-9]+]]:[[MAG_EXT_HI:[0-9]+]]{{\]}}, v[[MAG_EXT]]
; GCN: v_bfi_b32 v[[OUT_HI:[0-9]+]], s[[CONST]], v[[MAG_EXT_HI]], v[[SIGN_HI]]
; GCN: buffer_store_dwordx2 v{{\[}}[[MAG_EXT_LO]]:[[OUT_HI]]{{\]}}
; GCN: s_endpgm
define amdgpu_kernel void @test_copysign_out_f64_mag_f16_sign_f64(
  double addrspace(1)* %arg_out,
  half addrspace(1)* %arg_mag,
  double addrspace(1)* %arg_sign) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %arg_mag_gep = getelementptr half, half addrspace(1)* %arg_mag, i32 %tid
  %mag = load half, half addrspace(1)* %arg_mag_gep
  %mag.ext = fpext half %mag to double
  %arg_sign_gep = getelementptr double, double addrspace(1)* %arg_sign, i32 %tid
  %sign = load double, double addrspace(1)* %arg_sign_gep
  %out = call double @llvm.copysign.f64(double %mag.ext, double %sign)
  store double %out, double addrspace(1)* %arg_out
  ret void
}

; GCN-LABEL: {{^}}test_copysign_out_f32_mag_f32_sign_f16:
; GCN-DAG: {{buffer|flat|global}}_load_dword v[[MAG:[0-9]+]]
; GCN-DAG: {{buffer|flat|global}}_load_ushort v[[SIGN:[0-9]+]]
; GCN-DAG: s_brev_b32 s[[CONST:[0-9]+]], -2
; SI-DAG: v_cvt_f32_f16_e32 v[[SIGN_F32:[0-9]+]], v[[SIGN]]
; SI: v_bfi_b32 v[[OUT:[0-9]+]], s[[CONST]], v[[MAG]], v[[SIGN_F32]]
; GFX89-DAG: v_lshlrev_b32_e32 v[[SIGN_SHIFT:[0-9]+]], 16, v[[SIGN]]
; GFX89: v_bfi_b32 v[[OUT:[0-9]+]], s[[CONST]], v[[MAG]], v[[SIGN_SHIFT]]
; GCN: buffer_store_dword v[[OUT]]
; GCN: s_endpgm
define amdgpu_kernel void @test_copysign_out_f32_mag_f32_sign_f16(
  float addrspace(1)* %arg_out,
  float addrspace(1)* %arg_mag,
  half addrspace(1)* %arg_sign) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %arg_mag_gep = getelementptr float, float addrspace(1)* %arg_mag, i32 %tid
  %mag = load float, float addrspace(1)* %arg_mag_gep
  %arg_sign_gep = getelementptr half, half addrspace(1)* %arg_sign, i32 %tid
  %sign = load half, half addrspace(1)* %arg_sign_gep
  %sign.ext = fpext half %sign to float
  %out = call float @llvm.copysign.f32(float %mag, float %sign.ext)
  store float %out, float addrspace(1)* %arg_out
  ret void
}

; GCN-LABEL: {{^}}test_copysign_out_f64_mag_f64_sign_f16:
; GCN-DAG: {{buffer|flat|global}}_load_dwordx2 v{{\[}}[[MAG_LO:[0-9]+]]:[[MAG_HI:[0-9]+]]{{\]}}
; GCN-DAG: {{buffer|flat|global}}_load_ushort v[[SIGN:[0-9]+]]
; GCN-DAG: s_brev_b32 s[[CONST:[0-9]+]], -2
; SI-DAG: v_cvt_f32_f16_e32 v[[SIGN_F32:[0-9]+]], v[[SIGN]]
; SI: v_bfi_b32 v[[OUT_HI:[0-9]+]], s[[CONST]], v[[MAG_HI]], v[[SIGN_F32]]
; GFX89-DAG: v_lshlrev_b32_e32 v[[SIGN_SHIFT:[0-9]+]], 16, v[[SIGN]]
; GFX89: v_bfi_b32 v[[OUT_HI:[0-9]+]], s[[CONST]], v[[MAG_HI]], v[[SIGN_SHIFT]]
; GCN: buffer_store_dwordx2 v{{\[}}[[MAG_LO]]:[[OUT_HI]]{{\]}}
; GCN: s_endpgm
define amdgpu_kernel void @test_copysign_out_f64_mag_f64_sign_f16(
  double addrspace(1)* %arg_out,
  double addrspace(1)* %arg_mag,
  half addrspace(1)* %arg_sign) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %arg_mag_gep = getelementptr double, double addrspace(1)* %arg_mag, i32 %tid
  %mag = load double, double addrspace(1)* %arg_mag_gep
  %arg_sign_gep = getelementptr half, half addrspace(1)* %arg_sign, i32 %tid
  %sign = load half, half addrspace(1)* %arg_sign_gep
  %sign.ext = fpext half %sign to double
  %out = call double @llvm.copysign.f64(double %mag, double %sign.ext)
  store double %out, double addrspace(1)* %arg_out
  ret void
}

; GCN-LABEL: {{^}}test_copysign_out_f16_mag_f16_sign_f32:
; GCN-DAG: {{buffer|flat|global}}_load_ushort v[[MAG:[0-9]+]]
; GCN-DAG: {{buffer|flat|global}}_load_dword v[[SIGN:[0-9]+]]
; SI-DAG: s_brev_b32 s[[CONST:[0-9]+]], -2
; SI-DAG: v_cvt_f32_f16_e32 v[[MAG_F32:[0-9]+]], v[[MAG]]
; SI: v_bfi_b32 v[[OUT_F32:[0-9]+]], s[[CONST]], v[[MAG_F32]], v[[SIGN]]
; SI: v_cvt_f16_f32_e32 v[[OUT:[0-9]+]], v[[OUT_F32]]
; GFX89-DAG: s_movk_i32 s[[CONST:[0-9]+]], 0x7fff
; GFX89-DAG: v_lshrrev_b32_e32 v[[SIGN_SHIFT:[0-9]+]], 16, v[[SIGN]]
; GFX89: v_bfi_b32 v[[OUT:[0-9]+]], s[[CONST]], v[[MAG]], v[[SIGN_SHIFT]]
; GCN: buffer_store_short v[[OUT]]
; GCN: s_endpgm
define amdgpu_kernel void @test_copysign_out_f16_mag_f16_sign_f32(
  half addrspace(1)* %arg_out,
  half addrspace(1)* %arg_mag,
  float addrspace(1)* %arg_sign) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %arg_mag_gep = getelementptr half, half addrspace(1)* %arg_mag, i32 %tid
  %mag = load half, half addrspace(1)* %arg_mag_gep
  %arg_sign_gep = getelementptr float, float addrspace(1)* %arg_sign, i32 %tid
  %sign = load float, float addrspace(1)* %arg_sign_gep
  %sign.trunc = fptrunc float %sign to half
  %out = call half @llvm.copysign.f16(half %mag, half %sign.trunc)
  store half %out, half addrspace(1)* %arg_out
  ret void
}

; GCN-LABEL: {{^}}test_copysign_out_f16_mag_f16_sign_f64:
; GCN-DAG: {{buffer|flat|global}}_load_ushort v[[MAG:[0-9]+]]
; GCN-DAG: {{buffer|flat|global}}_load_dwordx2 v{{\[}}[[SIGN_LO:[0-9]+]]:[[SIGN_HI:[0-9]+]]{{\]}}
; SI-DAG: s_brev_b32 s[[CONST:[0-9]+]], -2
; SI-DAG: v_cvt_f32_f16_e32 v[[MAG_F32:[0-9]+]], v[[MAG]]
; SI: v_bfi_b32 v[[OUT_F32:[0-9]+]], s[[CONST]], v[[MAG_F32]], v[[SIGN_HI]]
; SI: v_cvt_f16_f32_e32 v[[OUT:[0-9]+]], v[[OUT_F32]]
; GFX89-DAG: s_movk_i32 s[[CONST:[0-9]+]], 0x7fff
; GFX89-DAG: v_lshrrev_b32_e32 v[[SIGN_SHIFT:[0-9]+]], 16, v[[SIGN_HI]]
; GFX89: v_bfi_b32 v[[OUT:[0-9]+]], s[[CONST]], v[[MAG]], v[[SIGN_SHIFT]]
; GCN: buffer_store_short v[[OUT]]
; GCN: s_endpgm
define amdgpu_kernel void @test_copysign_out_f16_mag_f16_sign_f64(
  half addrspace(1)* %arg_out,
  half addrspace(1)* %arg_mag,
  double addrspace(1)* %arg_sign) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %arg_mag_gep = getelementptr half, half addrspace(1)* %arg_mag, i32 %tid
  %mag = load half, half addrspace(1)* %arg_mag
  %arg_sign_gep = getelementptr double, double addrspace(1)* %arg_sign, i32 %tid
  %sign = load double, double addrspace(1)* %arg_sign_gep
  %sign.trunc = fptrunc double %sign to half
  %out = call half @llvm.copysign.f16(half %mag, half %sign.trunc)
  store half %out, half addrspace(1)* %arg_out
  ret void
}

; GCN-LABEL: {{^}}test_copysign_out_f16_mag_f32_sign_f16:
; GCN-DAG: {{buffer|flat|global}}_load_dword v[[MAG:[0-9]+]]
; GCN-DAG: {{buffer|flat|global}}_load_ushort v[[SIGN:[0-9]+]]
; SI-DAG: s_brev_b32 s[[CONST:[0-9]+]], -2
; SI-DAG: v_cvt_f16_f32_e32 v[[MAG_TRUNC:[0-9]+]], v[[MAG]]
; SI-DAG: v_cvt_f32_f16_e32 v[[SIGN_F32:[0-9]+]], v[[SIGN]]
; SI-DAG: v_cvt_f32_f16_e32 v[[MAG_F32:[0-9]+]], v[[MAG_TRUNC]]
; SI: v_bfi_b32 v[[OUT_F32:[0-9]+]], s[[CONST]], v[[MAG_F32]], v[[SIGN_F32]]
; SI: v_cvt_f16_f32_e32 v[[OUT:[0-9]+]], v[[OUT_F32]]
; GFX89-DAG: s_movk_i32 s[[CONST:[0-9]+]], 0x7fff
; GFX89-DAG: v_cvt_f16_f32_e32 v[[MAG_TRUNC:[0-9]+]], v[[MAG]]
; GFX89: v_bfi_b32 v[[OUT:[0-9]+]], s[[CONST]], v[[MAG_TRUNC]], v[[SIGN]]
; GCN: buffer_store_short v[[OUT]]
; GCN: s_endpgm
define amdgpu_kernel void @test_copysign_out_f16_mag_f32_sign_f16(
  half addrspace(1)* %arg_out,
  float addrspace(1)* %arg_mag,
  half addrspace(1)* %arg_sign) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %arg_mag_gep = getelementptr float, float addrspace(1)* %arg_mag, i32 %tid
  %mag = load float, float addrspace(1)* %arg_mag_gep
  %mag.trunc = fptrunc float %mag to half
  %arg_sign_gep = getelementptr half, half addrspace(1)* %arg_sign, i32 %tid
  %sign = load half, half addrspace(1)* %arg_sign_gep
  %out = call half @llvm.copysign.f16(half %mag.trunc, half %sign)
  store half %out, half addrspace(1)* %arg_out
  ret void
}

; GCN-LABEL: {{^}}test_copysign_out_f16_mag_f64_sign_f16:
; GCN: v_bfi_b32
; GCN: s_endpgm
define amdgpu_kernel void @test_copysign_out_f16_mag_f64_sign_f16(
  half addrspace(1)* %arg_out,
  double addrspace(1)* %arg_mag,
  half addrspace(1)* %arg_sign) {
entry:
  %mag = load double, double addrspace(1)* %arg_mag
  %mag.trunc = fptrunc double %mag to half
  %sign = load half, half addrspace(1)* %arg_sign
  %out = call half @llvm.copysign.f16(half %mag.trunc, half %sign)
  store half %out, half addrspace(1)* %arg_out
  ret void
}

; GCN-LABEL: {{^}}test_copysign_v2f16:
; GCN: v_bfi_b32
; GCN: v_bfi_b32
; VI: v_or_b32_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
; GCN: s_endpgm
define amdgpu_kernel void @test_copysign_v2f16(
  <2 x half> addrspace(1)* %arg_out,
  <2 x half> %arg_mag,
  <2 x half> %arg_sign) {
entry:
  %out = call <2 x half> @llvm.copysign.v2f16(<2 x half> %arg_mag, <2 x half> %arg_sign)
  store <2 x half> %out, <2 x half> addrspace(1)* %arg_out
  ret void
}

; GCN-LABEL: {{^}}test_copysign_v3f16:
; GCN: v_bfi_b32
; GCN: v_bfi_b32
; GCN: v_bfi_b32
; GCN: s_endpgm
define amdgpu_kernel void @test_copysign_v3f16(
  <3 x half> addrspace(1)* %arg_out,
  <3 x half> %arg_mag,
  <3 x half> %arg_sign) {
entry:
  %out = call <3 x half> @llvm.copysign.v3f16(<3 x half> %arg_mag, <3 x half> %arg_sign)
  store <3 x half> %out, <3 x half> addrspace(1)* %arg_out
  ret void
}

; GCN-LABEL: {{^}}test_copysign_v4f16:
; GCN: v_bfi_b32
; GCN: v_bfi_b32
; GCN: v_bfi_b32
; GCN: v_bfi_b32
; GCN: s_endpgm
define amdgpu_kernel void @test_copysign_v4f16(
  <4 x half> addrspace(1)* %arg_out,
  <4 x half> %arg_mag,
  <4 x half> %arg_sign) {
entry:
  %out = call <4 x half> @llvm.copysign.v4f16(<4 x half> %arg_mag, <4 x half> %arg_sign)
  store <4 x half> %out, <4 x half> addrspace(1)* %arg_out
  ret void
}
