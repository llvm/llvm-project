; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=VI %s

; Make sure we don't crash or assert on spir_kernel calling convention.

; GCN-LABEL: {{^}}kernel:
; GCN: s_endpgm
define spir_kernel void @kernel(i32 addrspace(1)* %out) {
entry:
  store i32 0, i32 addrspace(1)* %out
  ret void
}

; FIXME: This is treated like a kernel
; XGCN-LABEL: {{^}}func:
; XGCN: s_endpgm
; define spir_func void @func(i32 addrspace(1)* %out) {
; entry:
;   store i32 0, i32 addrspace(1)* %out
;   ret void
; }

; GCN-LABEL: {{^}}ps_ret_cc_f16:
; SI: v_cvt_f16_f32_e32 v0, v0
; SI: v_cvt_f32_f16_e32 v0, v0
; SI: v_add_f32_e32 v0, 1.0, v0

; VI: v_add_f16_e32 v0, 1.0, v0
; VI: ; return
define amdgpu_ps half @ps_ret_cc_f16(half %arg0) {
  %add = fadd half %arg0, 1.0
  ret half %add
}

; GCN-LABEL: {{^}}ps_ret_cc_inreg_f16:
; SI: v_cvt_f16_f32_e32 v0, s0
; SI: v_cvt_f32_f16_e32 v0, v0
; SI: v_add_f32_e32 v0, 1.0, v0

; VI: v_add_f16_e64 v0, s0, 1.0
; VI: ; return
define amdgpu_ps half @ps_ret_cc_inreg_f16(half inreg %arg0) {
  %add = fadd half %arg0, 1.0
  ret half %add
}

; GCN-LABEL: {{^}}fastcc:
; GCN: v_add_f32_e32 v0, 4.0, v0
define fastcc float @fastcc(float %arg0) #0 {
  %add = fadd float %arg0, 4.0
  ret float %add
}

; GCN-LABEL: {{^}}coldcc:
; GCN: v_add_f32_e32 v0, 4.0, v0
define coldcc float @coldcc(float %arg0) #0 {
 %add = fadd float %arg0, 4.0
 ret float %add
}

; GCN-LABEL: {{^}}call_coldcc:
; GCN: v_mov_b32_e32 v0, 1.0
; GCN: s_swappc_b64
define amdgpu_kernel void @call_coldcc() #0 {
  %val = call float @coldcc(float 1.0)
  store float %val, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}call_fastcc:
; GCN: v_mov_b32_e32 v0, 1.0
; GCN: s_swappc_b64
define amdgpu_kernel void @call_fastcc() #0 {
  %val = call float @fastcc(float 1.0)
  store float %val, float addrspace(1)* undef
  ret void
}

; Mesa compute shader: check for 47176 (COMPUTE_PGM_RSRC1) in .AMDGPU.config
; GCN-LABEL: .AMDGPU.config
; GCN: .long  47176
; GCN-LABEL: {{^}}cs_mesa:
define amdgpu_cs half @cs_mesa(half %arg0) {
  %add = fadd half %arg0, 1.0
  ret half %add
}

; Mesa pixel shader: check for 45096 (SPI_SHADER_PGM_RSRC1_PS) in .AMDGPU.config
; GCN-LABEL: .AMDGPU.config
; GCN: .long  45096
; GCN-LABEL: {{^}}ps_mesa_f16:
define amdgpu_ps half @ps_mesa_f16(half %arg0) {
  %add = fadd half %arg0, 1.0
  ret half %add
}

; Mesa vertex shader: check for 45352 (SPI_SHADER_PGM_RSRC1_VS) in .AMDGPU.config
; GCN-LABEL: .AMDGPU.config
; GCN: .long  45352
; GCN-LABEL: {{^}}vs_mesa:
define amdgpu_vs half @vs_mesa(half %arg0) {
  %add = fadd half %arg0, 1.0
  ret half %add
}

; Mesa geometry shader: check for 45608 (SPI_SHADER_PGM_RSRC1_GS) in .AMDGPU.config
; GCN-LABEL: .AMDGPU.config
; GCN: .long  45608
; GCN-LABEL: {{^}}gs_mesa:
define amdgpu_gs half @gs_mesa(half %arg0) {
  %add = fadd half %arg0, 1.0
  ret half %add
}

; Mesa hull shader: check for 46120 (SPI_SHADER_PGM_RSRC1_HS) in .AMDGPU.config
; GCN-LABEL: .AMDGPU.config
; GCN: .long  46120
; GCN-LABEL: {{^}}hs_mesa:
define amdgpu_hs half @hs_mesa(half %arg0) {
  %add = fadd half %arg0, 1.0
  ret half %add
}

; FIXME: Inconsistent ABI between targets
; GCN-LABEL: {{^}}ps_mesa_v2f16:
; VI: v_mov_b32_e32 v1, 0x3c00
; VI-NEXT: v_add_f16_sdwa v1, v0, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; VI-NEXT: v_add_f16_e32 v0, 1.0, v0
; VI-NEXT: v_or_b32_e32 v0, v0, v1
; VI-NEXT: ; return

; SI-DAG: v_cvt_f16_f32_e32 [[CVT_ELT0:v[0-9]+]], v0
; SI-DAG: v_cvt_f16_f32_e32 [[CVT_ELT1:v[0-9]+]], v1
; SI-DAG: v_cvt_f32_f16_e32 [[RECVT_ELT0:v[0-9]+]], [[CVT_ELT0]]
; SI-DAG: v_cvt_f32_f16_e32 [[RECVT_ELT1:v[0-9]+]], [[CVT_ELT1]]
; SI-DAG: v_add_f32_e32 v0, 1.0, [[RECVT_ELT0]]
; SI-DAG: v_add_f32_e32 v1, 1.0, [[RECVT_ELT1]]
; SI: ; return to shader part epilog
define amdgpu_ps <2 x half> @ps_mesa_v2f16(<2 x half> %arg0) {
  %add = fadd <2 x half> %arg0, <half 1.0, half 1.0>
  ret <2 x half> %add
}

; GCN-LABEL: {{^}}ps_mesa_inreg_v2f16:
; VI: s_lshr_b32 s1, s0, 16
; VI-NEXT: v_mov_b32_e32 v0, s1
; VI-NEXT: v_mov_b32_e32 v1, 0x3c00
; VI-NEXT: v_add_f16_sdwa v0, v0, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI-NEXT: v_add_f16_e64 v1, s0, 1.0
; VI-NEXT: v_or_b32_e32 v0, v1, v0
; VI-NEXT: ; return to shader part epilog

; SI-DAG: v_cvt_f16_f32_e32 [[CVT_ELT0:v[0-9]+]], s0
; SI-DAG: v_cvt_f16_f32_e32 [[CVT_ELT1:v[0-9]+]], s1
; SI-DAG: v_cvt_f32_f16_e32 [[RECVT_ELT0:v[0-9]+]], [[CVT_ELT0]]
; SI-DAG: v_cvt_f32_f16_e32 [[RECVT_ELT1:v[0-9]+]], [[CVT_ELT1]]
; SI-DAG: v_add_f32_e32 v0, 1.0, [[RECVT_ELT0]]
; SI-DAG: v_add_f32_e32 v1, 1.0, [[RECVT_ELT1]]
; SI: ; return to shader part epilog
define amdgpu_ps <2 x half> @ps_mesa_inreg_v2f16(<2 x half> inreg %arg0) {
  %add = fadd <2 x half> %arg0, <half 1.0, half 1.0>
  ret <2 x half> %add
}

; GCN-LABEL: {{^}}ps_mesa_v2i16:
; VI: v_mov_b32_e32 v2, 1
; VI: v_add_u16_e32 v1, 1, v0
; VI: v_add_u16_sdwa v0, v0, v2 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; VI: v_or_b32_e32 v0, v1, v0


; SI: v_lshlrev_b32_e32 v1, 16, v1
; SI: v_add_i32_e32 v0, vcc, 1, v0
; SI: v_add_i32_e32 v1, vcc, 0x10000, v1
; SI: v_and_b32
; SI: v_or_b32
define amdgpu_ps void @ps_mesa_v2i16(<2 x i16> %arg0) {
  %add = add <2 x i16> %arg0, <i16 1, i16 1>
  store <2 x i16> %add, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}ps_mesa_inreg_v2i16:
; VI: s_lshr_b32 s1, s0, 16
; VI: s_add_i32 s1, s1, 1
; VI: s_add_i32 s0, s0, 1
; VI: s_and_b32 s0, s0, 0xffff
; VI: s_lshl_b32 s1, s1, 16
; VI: s_or_b32 s0, s0, s1
; VI: v_mov_b32_e32 v0, s0

; SI: s_lshl_b32 s1, s1, 16
; SI: s_add_i32 s0, s0, 1
; SI: s_add_i32 s1, s1, 0x10000
; SI: s_and_b32 s0, s0, 0xffff
; SI: s_or_b32 s0, s0, s1
define amdgpu_ps void @ps_mesa_inreg_v2i16(<2 x i16> inreg %arg0) {
  %add = add <2 x i16> %arg0, <i16 1, i16 1>
  store <2 x i16> %add, <2 x i16> addrspace(1)* undef
  ret void
}

attributes #0 = { nounwind noinline }
