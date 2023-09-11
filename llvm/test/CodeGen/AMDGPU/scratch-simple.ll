; RUN: llc -mtriple=amdgcn -mtriple=amdgcn-- -mcpu=verde -amdgpu-use-divergent-register-indexing -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,SI,SIVI,MUBUF %s
; RUN: llc -mtriple=amdgcn -mtriple=amdgcn-- -mcpu=gfx803 -mattr=-flat-for-global -amdgpu-use-divergent-register-indexing -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,VI,SIVI,MUBUF %s
; RUN: llc -mtriple=amdgcn -mtriple=amdgcn-- -mcpu=gfx900 -mattr=-flat-for-global -amdgpu-use-divergent-register-indexing -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX9PLUS,MUBUF,GFX9-MUBUF,GFX9_10-MUBUF %s
; RUN: llc -mtriple=amdgcn -mtriple=amdgcn-- -mcpu=gfx900 -filetype=obj -amdgpu-use-divergent-register-indexing < %s | llvm-readobj -r - | FileCheck --check-prefix=RELS %s
; RUN: llc -mtriple=amdgcn -mtriple=amdgcn-- -mcpu=gfx1010 -mattr=-flat-for-global -amdgpu-use-divergent-register-indexing -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX9PLUS,MUBUF,GFX10_W32-MUBUF,GFX9_10-MUBUF %s
; RUN: llc -mtriple=amdgcn -mtriple=amdgcn-- -mcpu=gfx1010 -mattr=-flat-for-global,+wavefrontsize64 -amdgpu-use-divergent-register-indexing -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX9PLUS,MUBUF,GFX10_W64-MUBUF,GFX9_10-MUBUF %s
; RUN: llc -mtriple=amdgcn -mtriple=amdgcn-- -mcpu=gfx900 -mattr=-flat-for-global,+enable-flat-scratch -amdgpu-use-divergent-register-indexing -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX9PLUS,FLATSCR,GFX9-FLATSCR %s
; RUN: llc -mtriple=amdgcn -mtriple=amdgcn-- -mcpu=gfx1030 -mattr=-flat-for-global,+enable-flat-scratch -amdgpu-use-divergent-register-indexing -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX9PLUS,FLATSCR,GFX10-FLATSCR %s
; RUN: llc -mtriple=amdgcn -mtriple=amdgcn--amdpal -mcpu=gfx900 -mattr=-flat-for-global,+enable-flat-scratch -amdgpu-use-divergent-register-indexing -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX9PLUS,FLATSCR,GFX9-FLATSCR-PAL %s
; RUN: llc -mtriple=amdgcn -mtriple=amdgcn--amdpal -mcpu=gfx1030 -mattr=-flat-for-global,+enable-flat-scratch -amdgpu-use-divergent-register-indexing -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX9PLUS,FLATSCR,GFX10-FLATSCR-PAL %s
; RUN: llc -mtriple=amdgcn -mtriple=amdgcn-- -mcpu=gfx1100 -mattr=-flat-for-global,+enable-flat-scratch -amdgpu-use-divergent-register-indexing -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX9PLUS,GFX11-FLATSCR %s
; RUN: llc -mtriple=amdgcn -mtriple=amdgcn--amdpal -mcpu=gfx1100 -mattr=-flat-for-global,+enable-flat-scratch -amdgpu-use-divergent-register-indexing -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX9PLUS,GFX11-FLATSCR %s

; RELS: R_AMDGPU_ABS32_LO SCRATCH_RSRC_DWORD0
; RELS: R_AMDGPU_ABS32_LO SCRATCH_RSRC_DWORD1

; This used to fail due to a v_add_i32 instruction with an illegal immediate
; operand that was created during Local Stack Slot Allocation. Test case derived
; from https://bugs.freedesktop.org/show_bug.cgi?id=96602
;
; GCN-LABEL: {{^}}ps_main:

; GFX9-FLATSCR-DAG: s_add_u32 flat_scratch_lo, s0, s2
; GFX9-FLATSCR-DAG: s_addc_u32 flat_scratch_hi, s1, 0
; GFX9-FLATSCR-DAG: v_and_b32_e32 [[CLAMP_IDX:v[0-9]+]], 0x1fc, v0

; GFX10-FLATSCR: s_add_u32 s0, s0, s2
; GFX10-FLATSCR: s_addc_u32 s1, s1, 0
; GFX10-FLATSCR: s_setreg_b32 hwreg(HW_REG_FLAT_SCR_LO), s0
; GFX10-FLATSCR: s_setreg_b32 hwreg(HW_REG_FLAT_SCR_HI), s1

; GFX9-FLATSCR-PAL-DAG: s_getpc_b64 s[2:3]
; GFX9-FLATSCR-PAL-DAG: s_mov_b32 s2, s0
; GFX9-FLATSCR-PAL-DAG: s_load_dwordx2 s[2:3], s[2:3], 0x0
; GFX9-FLATSCR-PAL-DAG: v_lshlrev_b32_e32 v0, 2, v0
; GFX9-FLATSCR-PAL-DAG: v_mov_b32_e32 v0, 0xbf20e7f4
; GFX9-FLATSCR-PAL-DAG: s_mov_b32 s0, 0
; GFX9-FLATSCR-PAL-DAG: s_waitcnt lgkmcnt(0)
; GFX9-FLATSCR-PAL-DAG: s_and_b32 s3, s3, 0xffff
; GFX9-FLATSCR-PAL-DAG: s_add_u32 flat_scratch_lo, s2, s0
; GFX9-FLATSCR-PAL-DAG: s_addc_u32 flat_scratch_hi, s3, 0
; GFX9-FLATSCR-PAL-DAG: v_and_b32_e32 [[CLAMP_IDX:v[0-9]+]], 0x1fc, v0

; GFX10-FLATSCR-PAL: s_getpc_b64 s[2:3]
; GFX10-FLATSCR-PAL: s_mov_b32 s2, s0
; GFX10-FLATSCR-PAL: s_load_dwordx2 s[2:3], s[2:3], 0x0
; GFX10-FLATSCR-PAL: s_waitcnt lgkmcnt(0)
; GFX10-FLATSCR-PAL: s_and_b32 s3, s3, 0xffff
; GFX10-FLATSCR-PAL: s_add_u32 s2, s2, s0
; GFX10-FLATSCR-PAL: s_addc_u32 s3, s3, 0
; GFX10-FLATSCR-PAL: s_setreg_b32 hwreg(HW_REG_FLAT_SCR_LO), s2
; GFX10-FLATSCR-PAL: s_setreg_b32 hwreg(HW_REG_FLAT_SCR_HI), s3

; MUBUF-DAG: s_mov_b32 s0, SCRATCH_RSRC_DWORD0
; MUBUF-DAG: s_mov_b32 s1, SCRATCH_RSRC_DWORD1
; MUBUF-DAG: s_mov_b32 s2, -1
; SI-DAG: s_mov_b32 s3, 0xe8f000
; VI-DAG: s_mov_b32 s3, 0xe80000
; GFX9-MUBUF-DAG: s_mov_b32 s3, 0xe00000
; GFX10_W32-MUBUF-DAG: s_mov_b32 s3, 0x31c16000
; GFX10_W64-MUBUF-DAG: s_mov_b32 s3, 0x31e16000

; FLATSCR-NOT: SCRATCH_RSRC_DWORD

; GFX9-FLATSCR: s_mov_b32 [[SP:[^,]+]], 0
; GFX9-FLATSCR: scratch_store_dwordx4 off, v[{{[0-9:]+}}], [[SP]] offset:

; GFX10-FLATSCR: scratch_store_dwordx4 off, v[{{[0-9:]+}}], off offset:

; MUBUF-DAG:     v_lshlrev_b32_e32 [[BYTES:v[0-9]+]], 2, v0
; MUBUF-DAG:     v_and_b32_e32 [[CLAMP_IDX:v[0-9]+]], 0x1fc, [[BYTES]]
; GFX10-FLATSCR: v_and_b32_e32 [[CLAMP_IDX:v[0-9]+]], 0x1fc, v0
; GFX10-FLATSCR-PAL: v_and_b32_e32 [[CLAMP_IDX:v[0-9]+]], 0x1fc, v0
; GFX11-FLATSCR: v_and_b32_e32 [[CLAMP_IDX:v[0-9]+]], 0x1fc, v0

; MUBUF-DAG: v_add{{_|_nc_}}{{i|u}}32_e32 [[HI_OFF:v[0-9]+]],{{.*}} 0x280, [[CLAMP_IDX]]
; MUBUF-DAG: v_add{{_|_nc_}}{{i|u}}32_e32 [[LO_OFF:v[0-9]+]],{{.*}} {{v2|0x80}}, [[CLAMP_IDX]]
; FLATSCR: v_add{{_|_nc_}}{{u32|b32}}_e32 [[LO_OFF:v[0-9]+]],{{.*}} {{v2|0x80}}, [[CLAMP_IDX]]

; MUBUF: buffer_load_dword {{v[0-9]+}}, [[LO_OFF]], {{s\[[0-9]+:[0-9]+\]}}, 0 offen
; MUBUF: buffer_load_dword {{v[0-9]+}}, [[HI_OFF]], {{s\[[0-9]+:[0-9]+\]}}, 0 offen
; FLATSCR: scratch_load_dword {{v[0-9]+}}, [[LO_OFF]], off
; GFX11-FLATSCR: scratch_load_b32 {{v[0-9]+}}, [[CLAMP_IDX]], off offset:128
define amdgpu_ps float @ps_main(i32 %idx) {
  %v1 = extractelement <81 x float> <float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float 0x3FE41CFEA0000000, float 0xBFE7A693C0000000, float 0xBFEA477C60000000, float 0xBFEBE5DC60000000, float 0xBFEC71C720000000, float 0xBFEBE5DC60000000, float 0xBFEA477C60000000, float 0xBFE7A693C0000000, float 0xBFE41CFEA0000000, float 0x3FDF9B13E0000000, float 0x3FDF9B1380000000, float 0x3FD5C53B80000000, float 0x3FD5C53B00000000, float 0x3FC6326AC0000000, float 0x3FC63269E0000000, float 0xBEE05CEB00000000, float 0xBEE086A320000000, float 0xBFC63269E0000000, float 0xBFC6326AC0000000, float 0xBFD5C53B80000000, float 0xBFD5C53B80000000, float 0xBFDF9B13E0000000, float 0xBFDF9B1460000000, float 0xBFE41CFE80000000, float 0x3FE7A693C0000000, float 0x3FEA477C20000000, float 0x3FEBE5DC40000000, float 0x3FEC71C6E0000000, float 0x3FEBE5DC40000000, float 0x3FEA477C20000000, float 0x3FE7A693C0000000, float 0xBFE41CFE80000000>, i32 %idx
  %v2 = extractelement <81 x float> <float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float 0xBFE41CFEA0000000, float 0xBFDF9B13E0000000, float 0xBFD5C53B80000000, float 0xBFC6326AC0000000, float 0x3EE0789320000000, float 0x3FC6326AC0000000, float 0x3FD5C53B80000000, float 0x3FDF9B13E0000000, float 0x3FE41CFEA0000000, float 0xBFE7A693C0000000, float 0x3FE7A693C0000000, float 0xBFEA477C20000000, float 0x3FEA477C20000000, float 0xBFEBE5DC40000000, float 0x3FEBE5DC40000000, float 0xBFEC71C720000000, float 0x3FEC71C6E0000000, float 0xBFEBE5DC60000000, float 0x3FEBE5DC40000000, float 0xBFEA477C20000000, float 0x3FEA477C20000000, float 0xBFE7A693C0000000, float 0x3FE7A69380000000, float 0xBFE41CFEA0000000, float 0xBFDF9B13E0000000, float 0xBFD5C53B80000000, float 0xBFC6326AC0000000, float 0x3EE0789320000000, float 0x3FC6326AC0000000, float 0x3FD5C53B80000000, float 0x3FDF9B13E0000000, float 0x3FE41CFE80000000>, i32 %idx
  %r = fadd float %v1, %v2
  ret float %r
}

; GCN-LABEL: {{^}}vs_main:
; GFX9-FLATSCR: s_add_u32 flat_scratch_lo, s0, s2
; GFX9-FLATSCR: s_addc_u32 flat_scratch_hi, s1, 0

; GFX10-FLATSCR: s_add_u32 s0, s0, s2
; GFX10-FLATSCR: s_addc_u32 s1, s1, 0
; GFX10-FLATSCR: s_setreg_b32 hwreg(HW_REG_FLAT_SCR_LO), s0
; GFX10-FLATSCR: s_setreg_b32 hwreg(HW_REG_FLAT_SCR_HI), s1

; GFX9-FLATSCR-PAL-DAG: s_getpc_b64 s[2:3]
; GFX9-FLATSCR-PAL-DAG: s_mov_b32 s2, s0
; GFX9-FLATSCR-PAL-DAG: s_load_dwordx2 s[2:3], s[2:3], 0x0
; GFX9-FLATSCR-PAL-DAG: v_lshlrev_b32_e32 v0, 2, v0
; GFX9-FLATSCR-PAL-DAG: v_mov_b32_e32 v0, 0xbf20e7f4
; GFX9-FLATSCR-PAL-DAG: s_mov_b32 s0, 0
; GFX9-FLATSCR-PAL-DAG: s_waitcnt lgkmcnt(0)
; GFX9-FLATSCR-PAL-DAG: s_and_b32 s3, s3, 0xffff
; GFX9-FLATSCR-PAL-DAG: s_add_u32 flat_scratch_lo, s2, s0
; GFX9-FLATSCR-PAL-DAG: s_addc_u32 flat_scratch_hi, s3, 0

; GFX10-FLATSCR-PAL: s_getpc_b64 s[2:3]
; GFX10-FLATSCR-PAL: s_mov_b32 s2, s0
; GFX10-FLATSCR-PAL: s_load_dwordx2 s[2:3], s[2:3], 0x0
; GFX10-FLATSCR-PAL: s_waitcnt lgkmcnt(0)
; GFX10-FLATSCR-PAL: s_and_b32 s3, s3, 0xffff
; GFX10-FLATSCR-PAL: s_add_u32 s2, s2, s0
; GFX10-FLATSCR-PAL: s_addc_u32 s3, s3, 0
; GFX10-FLATSCR-PAL: s_setreg_b32 hwreg(HW_REG_FLAT_SCR_LO), s2
; GFX10-FLATSCR-PAL: s_setreg_b32 hwreg(HW_REG_FLAT_SCR_HI), s3

; MUBUF-DAG: s_mov_b32 s0, SCRATCH_RSRC_DWORD0

; FLATSCR-NOT: SCRATCH_RSRC_DWORD

; MUBUF: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 offen
; MUBUF: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 offen

; GFX9-FLATSCR: s_mov_b32 [[SP:[^,]+]], 0
; GFX9-FLATSCR: scratch_store_dwordx4 off, v[{{[0-9:]+}}], [[SP]] offset:

; FLATSCR: scratch_load_dword {{v[0-9]+}}, {{v[0-9]+}}, off
; FLATSCR: scratch_load_dword {{v[0-9]+}}, {{v[0-9]+}}, off

; GFX11-FLATSCR: scratch_load_b32 {{v[0-9]+}}, {{v[0-9]+}}, off
; GFX11-FLATSCR: scratch_load_b32 {{v[0-9]+}}, {{v[0-9]+}}, off

define amdgpu_vs float @vs_main(i32 %idx) {
  %v1 = extractelement <81 x float> <float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float 0x3FE41CFEA0000000, float 0xBFE7A693C0000000, float 0xBFEA477C60000000, float 0xBFEBE5DC60000000, float 0xBFEC71C720000000, float 0xBFEBE5DC60000000, float 0xBFEA477C60000000, float 0xBFE7A693C0000000, float 0xBFE41CFEA0000000, float 0x3FDF9B13E0000000, float 0x3FDF9B1380000000, float 0x3FD5C53B80000000, float 0x3FD5C53B00000000, float 0x3FC6326AC0000000, float 0x3FC63269E0000000, float 0xBEE05CEB00000000, float 0xBEE086A320000000, float 0xBFC63269E0000000, float 0xBFC6326AC0000000, float 0xBFD5C53B80000000, float 0xBFD5C53B80000000, float 0xBFDF9B13E0000000, float 0xBFDF9B1460000000, float 0xBFE41CFE80000000, float 0x3FE7A693C0000000, float 0x3FEA477C20000000, float 0x3FEBE5DC40000000, float 0x3FEC71C6E0000000, float 0x3FEBE5DC40000000, float 0x3FEA477C20000000, float 0x3FE7A693C0000000, float 0xBFE41CFE80000000>, i32 %idx
  %v2 = extractelement <81 x float> <float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float 0xBFE41CFEA0000000, float 0xBFDF9B13E0000000, float 0xBFD5C53B80000000, float 0xBFC6326AC0000000, float 0x3EE0789320000000, float 0x3FC6326AC0000000, float 0x3FD5C53B80000000, float 0x3FDF9B13E0000000, float 0x3FE41CFEA0000000, float 0xBFE7A693C0000000, float 0x3FE7A693C0000000, float 0xBFEA477C20000000, float 0x3FEA477C20000000, float 0xBFEBE5DC40000000, float 0x3FEBE5DC40000000, float 0xBFEC71C720000000, float 0x3FEC71C6E0000000, float 0xBFEBE5DC60000000, float 0x3FEBE5DC40000000, float 0xBFEA477C20000000, float 0x3FEA477C20000000, float 0xBFE7A693C0000000, float 0x3FE7A69380000000, float 0xBFE41CFEA0000000, float 0xBFDF9B13E0000000, float 0xBFD5C53B80000000, float 0xBFC6326AC0000000, float 0x3EE0789320000000, float 0x3FC6326AC0000000, float 0x3FD5C53B80000000, float 0x3FDF9B13E0000000, float 0x3FE41CFE80000000>, i32 %idx
  %r = fadd float %v1, %v2
  ret float %r
}

; GCN-LABEL: {{^}}cs_main:
; GFX9-FLATSCR: s_add_u32 flat_scratch_lo, s0, s2
; GFX9-FLATSCR: s_addc_u32 flat_scratch_hi, s1, 0

; GFX10-FLATSCR: s_add_u32 s0, s0, s2
; GFX10-FLATSCR: s_addc_u32 s1, s1, 0
; GFX10-FLATSCR: s_setreg_b32 hwreg(HW_REG_FLAT_SCR_LO), s0
; GFX10-FLATSCR: s_setreg_b32 hwreg(HW_REG_FLAT_SCR_HI), s1

; GFX9-FLATSCR-PAL-DAG: s_getpc_b64 s[2:3]
; GFX9-FLATSCR-PAL-DAG: s_mov_b32 s2, s0
; GFX9-FLATSCR-PAL-DAG: s_load_dwordx2 s[2:3], s[2:3], 0x10
; GFX9-FLATSCR-PAL-DAG: v_lshlrev_b32_e32 v0, 2, v0
; GFX9-FLATSCR-PAL-DAG: v_mov_b32_e32 v0, 0xbf20e7f4
; GFX9-FLATSCR-PAL-DAG: s_mov_b32 s0, 0
; GFX9-FLATSCR-PAL-DAG: s_waitcnt lgkmcnt(0)
; GFX9-FLATSCR-PAL-DAG: s_and_b32 s3, s3, 0xffff
; GFX9-FLATSCR-PAL-DAG: s_add_u32 flat_scratch_lo, s2, s0
; GFX9-FLATSCR-PAL-DAG: s_addc_u32 flat_scratch_hi, s3, 0

; GFX10-FLATSCR-PAL: s_getpc_b64 s[2:3]
; GFX10-FLATSCR-PAL: s_mov_b32 s2, s0
; GFX10-FLATSCR-PAL: s_load_dwordx2 s[2:3], s[2:3], 0x10
; GFX10-FLATSCR-PAL: s_waitcnt lgkmcnt(0)
; GFX10-FLATSCR-PAL: s_and_b32 s3, s3, 0xffff
; GFX10-FLATSCR-PAL: s_add_u32 s2, s2, s0
; GFX10-FLATSCR-PAL: s_addc_u32 s3, s3, 0
; GFX10-FLATSCR-PAL: s_setreg_b32 hwreg(HW_REG_FLAT_SCR_LO), s2
; GFX10-FLATSCR-PAL: s_setreg_b32 hwreg(HW_REG_FLAT_SCR_HI), s3

; MUBUF-DAG: s_mov_b32 s0, SCRATCH_RSRC_DWORD0

; FLATSCR-NOT: SCRATCH_RSRC_DWORD

; MUBUF: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 offen
; MUBUF: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 offen

; FLATSCR: scratch_load_dword {{v[0-9]+}}, {{v[0-9]+}}, off
; FLATSCR: scratch_load_dword {{v[0-9]+}}, {{v[0-9]+}}, off

; GFX11-FLATSCR: scratch_load_b32 {{v[0-9]+}}, {{v[0-9]+}}, off
; GFX11-FLATSCR: scratch_load_b32 {{v[0-9]+}}, {{v[0-9]+}}, off

define amdgpu_cs float @cs_main(i32 %idx) {
  %v1 = extractelement <81 x float> <float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float 0x3FE41CFEA0000000, float 0xBFE7A693C0000000, float 0xBFEA477C60000000, float 0xBFEBE5DC60000000, float 0xBFEC71C720000000, float 0xBFEBE5DC60000000, float 0xBFEA477C60000000, float 0xBFE7A693C0000000, float 0xBFE41CFEA0000000, float 0x3FDF9B13E0000000, float 0x3FDF9B1380000000, float 0x3FD5C53B80000000, float 0x3FD5C53B00000000, float 0x3FC6326AC0000000, float 0x3FC63269E0000000, float 0xBEE05CEB00000000, float 0xBEE086A320000000, float 0xBFC63269E0000000, float 0xBFC6326AC0000000, float 0xBFD5C53B80000000, float 0xBFD5C53B80000000, float 0xBFDF9B13E0000000, float 0xBFDF9B1460000000, float 0xBFE41CFE80000000, float 0x3FE7A693C0000000, float 0x3FEA477C20000000, float 0x3FEBE5DC40000000, float 0x3FEC71C6E0000000, float 0x3FEBE5DC40000000, float 0x3FEA477C20000000, float 0x3FE7A693C0000000, float 0xBFE41CFE80000000>, i32 %idx
  %v2 = extractelement <81 x float> <float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float 0xBFE41CFEA0000000, float 0xBFDF9B13E0000000, float 0xBFD5C53B80000000, float 0xBFC6326AC0000000, float 0x3EE0789320000000, float 0x3FC6326AC0000000, float 0x3FD5C53B80000000, float 0x3FDF9B13E0000000, float 0x3FE41CFEA0000000, float 0xBFE7A693C0000000, float 0x3FE7A693C0000000, float 0xBFEA477C20000000, float 0x3FEA477C20000000, float 0xBFEBE5DC40000000, float 0x3FEBE5DC40000000, float 0xBFEC71C720000000, float 0x3FEC71C6E0000000, float 0xBFEBE5DC60000000, float 0x3FEBE5DC40000000, float 0xBFEA477C20000000, float 0x3FEA477C20000000, float 0xBFE7A693C0000000, float 0x3FE7A69380000000, float 0xBFE41CFEA0000000, float 0xBFDF9B13E0000000, float 0xBFD5C53B80000000, float 0xBFC6326AC0000000, float 0x3EE0789320000000, float 0x3FC6326AC0000000, float 0x3FD5C53B80000000, float 0x3FDF9B13E0000000, float 0x3FE41CFE80000000>, i32 %idx
  %r = fadd float %v1, %v2
  ret float %r
}

; GCN-LABEL: {{^}}hs_main:
; GFX9-FLATSCR: s_add_u32 flat_scratch_lo, s0, s5
; GFX9-FLATSCR: s_addc_u32 flat_scratch_hi, s1, 0

; GFX10-FLATSCR: s_add_u32 s0, s0, s5
; GFX10-FLATSCR: s_addc_u32 s1, s1, 0
; GFX10-FLATSCR: s_setreg_b32 hwreg(HW_REG_FLAT_SCR_LO), s0
; GFX10-FLATSCR: s_setreg_b32 hwreg(HW_REG_FLAT_SCR_HI), s1

; SIVI: s_mov_b32 s0, SCRATCH_RSRC_DWORD0
; SIVI-NOT: s_mov_b32 s0
; SIVI: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 offen
; SIVI: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 offen

; GFX9_10-MUBUF: s_mov_b32 s0, SCRATCH_RSRC_DWORD0
; GFX9PLUS-NOT:   s_mov_b32 s5
; GFX9_10-MUBUF: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 offen
; GFX9_10-MUBUF: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 offen

; FLATSCR-NOT: SCRATCH_RSRC_DWORD
; FLATSCR: scratch_load_dword {{v[0-9]+}}, {{v[0-9]+}}, off
; FLATSCR: scratch_load_dword {{v[0-9]+}}, {{v[0-9]+}}, off

; GFX11-FLATSCR: scratch_load_b32 {{v[0-9]+}}, {{v[0-9]+}}, off
; GFX11-FLATSCR: scratch_load_b32 {{v[0-9]+}}, {{v[0-9]+}}, off
define amdgpu_hs float @hs_main(i32 %idx) {
  %v1 = extractelement <81 x float> <float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float 0x3FE41CFEA0000000, float 0xBFE7A693C0000000, float 0xBFEA477C60000000, float 0xBFEBE5DC60000000, float 0xBFEC71C720000000, float 0xBFEBE5DC60000000, float 0xBFEA477C60000000, float 0xBFE7A693C0000000, float 0xBFE41CFEA0000000, float 0x3FDF9B13E0000000, float 0x3FDF9B1380000000, float 0x3FD5C53B80000000, float 0x3FD5C53B00000000, float 0x3FC6326AC0000000, float 0x3FC63269E0000000, float 0xBEE05CEB00000000, float 0xBEE086A320000000, float 0xBFC63269E0000000, float 0xBFC6326AC0000000, float 0xBFD5C53B80000000, float 0xBFD5C53B80000000, float 0xBFDF9B13E0000000, float 0xBFDF9B1460000000, float 0xBFE41CFE80000000, float 0x3FE7A693C0000000, float 0x3FEA477C20000000, float 0x3FEBE5DC40000000, float 0x3FEC71C6E0000000, float 0x3FEBE5DC40000000, float 0x3FEA477C20000000, float 0x3FE7A693C0000000, float 0xBFE41CFE80000000>, i32 %idx
  %v2 = extractelement <81 x float> <float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float 0xBFE41CFEA0000000, float 0xBFDF9B13E0000000, float 0xBFD5C53B80000000, float 0xBFC6326AC0000000, float 0x3EE0789320000000, float 0x3FC6326AC0000000, float 0x3FD5C53B80000000, float 0x3FDF9B13E0000000, float 0x3FE41CFEA0000000, float 0xBFE7A693C0000000, float 0x3FE7A693C0000000, float 0xBFEA477C20000000, float 0x3FEA477C20000000, float 0xBFEBE5DC40000000, float 0x3FEBE5DC40000000, float 0xBFEC71C720000000, float 0x3FEC71C6E0000000, float 0xBFEBE5DC60000000, float 0x3FEBE5DC40000000, float 0xBFEA477C20000000, float 0x3FEA477C20000000, float 0xBFE7A693C0000000, float 0x3FE7A69380000000, float 0xBFE41CFEA0000000, float 0xBFDF9B13E0000000, float 0xBFD5C53B80000000, float 0xBFC6326AC0000000, float 0x3EE0789320000000, float 0x3FC6326AC0000000, float 0x3FD5C53B80000000, float 0x3FDF9B13E0000000, float 0x3FE41CFE80000000>, i32 %idx
  %r = fadd float %v1, %v2
  ret float %r
}

; GCN-LABEL: {{^}}gs_main:
; GFX9-FLATSCR: s_add_u32 flat_scratch_lo, s0, s5
; GFX9-FLATSCR: s_addc_u32 flat_scratch_hi, s1, 0

; GFX10-FLATSCR: s_add_u32 s0, s0, s5
; GFX10-FLATSCR: s_addc_u32 s1, s1, 0
; GFX10-FLATSCR: s_setreg_b32 hwreg(HW_REG_FLAT_SCR_LO), s0
; GFX10-FLATSCR: s_setreg_b32 hwreg(HW_REG_FLAT_SCR_HI), s1

; GFX9-FLATSCR-PAL-DAG: s_getpc_b64 s[0:1]
; GFX9-FLATSCR-PAL-DAG: s_mov_b32 s0, s8
; GFX9-FLATSCR-PAL-DAG: s_load_dwordx2 s[0:1], s[0:1], 0x0
; GFX9-FLATSCR-PAL-DAG: v_lshlrev_b32_e32 v0, 2, v0
; GFX9-FLATSCR-PAL-DAG: v_mov_b32_e32 v0, 0xbf20e7f4
; GFX9-FLATSCR-PAL-DAG: s_mov_b32 s0, 0
; GFX9-FLATSCR-PAL-DAG: s_waitcnt lgkmcnt(0)
; GFX9-FLATSCR-PAL-DAG: s_and_b32 s1, s1, 0xffff
; GFX9-FLATSCR-PAL-DAG: s_add_u32 flat_scratch_lo, s0, s5
; GFX9-FLATSCR-PAL-DAG: s_addc_u32 flat_scratch_hi, s1, 0

; GFX10-FLATSCR-PAL: s_getpc_b64 s[0:1]
; GFX10-FLATSCR-PAL: s_mov_b32 s0, s8
; GFX10-FLATSCR-PAL: s_load_dwordx2 s[0:1], s[0:1], 0x0
; GFX10-FLATSCR-PAL: s_waitcnt lgkmcnt(0)
; GFX10-FLATSCR-PAL: s_and_b32 s1, s1, 0xffff
; GFX10-FLATSCR-PAL: s_add_u32 s0, s0, s5
; GFX10-FLATSCR-PAL: s_addc_u32 s1, s1, 0
; GFX10-FLATSCR-PAL: s_setreg_b32 hwreg(HW_REG_FLAT_SCR_LO), s0
; GFX10-FLATSCR-PAL: s_setreg_b32 hwreg(HW_REG_FLAT_SCR_HI), s1

; SIVI: s_mov_b32 s0, SCRATCH_RSRC_DWORD0
; SIVI: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 offen
; SIVI: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 offen

; GFX9_10-MUBUF: s_mov_b32 s0, SCRATCH_RSRC_DWORD0
; GFX9_10-MUBUF: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 offen
; GFX9_10-MUBUF: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 offen

; FLATSCR-NOT: SCRATCH_RSRC_DWORD
; FLATSCR: scratch_load_dword {{v[0-9]+}}, {{v[0-9]+}}, off
; FLATSCR: scratch_load_dword {{v[0-9]+}}, {{v[0-9]+}}, off

; GFX11-FLATSCR: scratch_load_b32 {{v[0-9]+}}, {{v[0-9]+}}, off
; GFX11-FLATSCR: scratch_load_b32 {{v[0-9]+}}, {{v[0-9]+}}, off
define amdgpu_gs float @gs_main(i32 %idx) {
  %v1 = extractelement <81 x float> <float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float 0x3FE41CFEA0000000, float 0xBFE7A693C0000000, float 0xBFEA477C60000000, float 0xBFEBE5DC60000000, float 0xBFEC71C720000000, float 0xBFEBE5DC60000000, float 0xBFEA477C60000000, float 0xBFE7A693C0000000, float 0xBFE41CFEA0000000, float 0x3FDF9B13E0000000, float 0x3FDF9B1380000000, float 0x3FD5C53B80000000, float 0x3FD5C53B00000000, float 0x3FC6326AC0000000, float 0x3FC63269E0000000, float 0xBEE05CEB00000000, float 0xBEE086A320000000, float 0xBFC63269E0000000, float 0xBFC6326AC0000000, float 0xBFD5C53B80000000, float 0xBFD5C53B80000000, float 0xBFDF9B13E0000000, float 0xBFDF9B1460000000, float 0xBFE41CFE80000000, float 0x3FE7A693C0000000, float 0x3FEA477C20000000, float 0x3FEBE5DC40000000, float 0x3FEC71C6E0000000, float 0x3FEBE5DC40000000, float 0x3FEA477C20000000, float 0x3FE7A693C0000000, float 0xBFE41CFE80000000>, i32 %idx
  %v2 = extractelement <81 x float> <float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float 0xBFE41CFEA0000000, float 0xBFDF9B13E0000000, float 0xBFD5C53B80000000, float 0xBFC6326AC0000000, float 0x3EE0789320000000, float 0x3FC6326AC0000000, float 0x3FD5C53B80000000, float 0x3FDF9B13E0000000, float 0x3FE41CFEA0000000, float 0xBFE7A693C0000000, float 0x3FE7A693C0000000, float 0xBFEA477C20000000, float 0x3FEA477C20000000, float 0xBFEBE5DC40000000, float 0x3FEBE5DC40000000, float 0xBFEC71C720000000, float 0x3FEC71C6E0000000, float 0xBFEBE5DC60000000, float 0x3FEBE5DC40000000, float 0xBFEA477C20000000, float 0x3FEA477C20000000, float 0xBFE7A693C0000000, float 0x3FE7A69380000000, float 0xBFE41CFEA0000000, float 0xBFDF9B13E0000000, float 0xBFD5C53B80000000, float 0xBFC6326AC0000000, float 0x3EE0789320000000, float 0x3FC6326AC0000000, float 0x3FD5C53B80000000, float 0x3FDF9B13E0000000, float 0x3FE41CFE80000000>, i32 %idx
  %r = fadd float %v1, %v2
  ret float %r
}

; Mesa GS and HS shaders have the preloaded scratch wave offset SGPR fixed at
; SGPR5, and the inreg implementation is used to reference it in the IR. The
; following tests confirm the shader and anything inserted after the return
; (i.e. SI_RETURN_TO_EPILOG) can access the scratch wave offset.

; GCN-LABEL: {{^}}hs_ir_uses_scratch_offset:
; GFX9-FLATSCR: s_add_u32 flat_scratch_lo, s0, s5
; GFX9-FLATSCR: s_addc_u32 flat_scratch_hi, s1, 0

; GFX10-FLATSCR: s_add_u32 s0, s0, s5
; GFX10-FLATSCR: s_addc_u32 s1, s1, 0
; GFX10-FLATSCR: s_setreg_b32 hwreg(HW_REG_FLAT_SCR_LO), s0
; GFX10-FLATSCR: s_setreg_b32 hwreg(HW_REG_FLAT_SCR_HI), s1

; GFX9-FLATSCR-PAL-DAG: s_getpc_b64 s[0:1]
; GFX9-FLATSCR-PAL-DAG: s_mov_b32 s0, s8
; GFX9-FLATSCR-PAL-DAG: s_load_dwordx2 s[0:1], s[0:1], 0x0
; GFX9-FLATSCR-PAL-DAG: v_lshlrev_b32_e32 v0, 2, v0
; GFX9-FLATSCR-PAL-DAG: v_mov_b32_e32 v0, 0xbf20e7f4
; GFX9-FLATSCR-PAL-DAG: s_mov_b32 s0, 0
; GFX9-FLATSCR-PAL-DAG: s_waitcnt lgkmcnt(0)
; GFX9-FLATSCR-PAL-DAG: s_and_b32 s1, s1, 0xffff
; GFX9-FLATSCR-PAL-DAG: s_add_u32 flat_scratch_lo, s0, s5
; GFX9-FLATSCR-PAL-DAG: s_addc_u32 flat_scratch_hi, s1, 0

; GFX10-FLATSCR-PAL: s_getpc_b64 s[0:1]
; GFX10-FLATSCR-PAL: s_mov_b32 s0, s8
; GFX10-FLATSCR-PAL: s_load_dwordx2 s[0:1], s[0:1], 0x0
; GFX10-FLATSCR-PAL: s_waitcnt lgkmcnt(0)
; GFX10-FLATSCR-PAL: s_and_b32 s1, s1, 0xffff
; GFX10-FLATSCR-PAL: s_add_u32 s0, s0, s5
; GFX10-FLATSCR-PAL: s_addc_u32 s1, s1, 0
; GFX10-FLATSCR-PAL: s_setreg_b32 hwreg(HW_REG_FLAT_SCR_LO), s0
; GFX10-FLATSCR-PAL: s_setreg_b32 hwreg(HW_REG_FLAT_SCR_HI), s1

; MUBUF: s_mov_b32 s8, SCRATCH_RSRC_DWORD0
; FLATSCR-NOT: SCRATCH_RSRC_DWORD

; SIVI-NOT: s_mov_b32 s6
; SIVI: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 offen
; SIVI: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 offen

; GFX9_10-MUBUF-DAG: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 offen
; GFX9_10-MUBUF-DAG: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 offen

; MUBUF-DAG: s_mov_b32 s2, s5

; FLATSCR-DAG: scratch_load_dword {{v[0-9]+}}, {{v[0-9]+}}, off
; FLATSCR-DAG: scratch_load_dword {{v[0-9]+}}, {{v[0-9]+}}, off

; GFX11-FLATSCR: scratch_load_b32 {{v[0-9]+}}, {{v[0-9]+}}, off
; GFX11-FLATSCR: scratch_load_b32 {{v[0-9]+}}, {{v[0-9]+}}, off
define amdgpu_hs <{i32, i32, i32, float}> @hs_ir_uses_scratch_offset(i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg %swo, i32 %idx) {
  %v1 = extractelement <81 x float> <float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float 0x3FE41CFEA0000000, float 0xBFE7A693C0000000, float 0xBFEA477C60000000, float 0xBFEBE5DC60000000, float 0xBFEC71C720000000, float 0xBFEBE5DC60000000, float 0xBFEA477C60000000, float 0xBFE7A693C0000000, float 0xBFE41CFEA0000000, float 0x3FDF9B13E0000000, float 0x3FDF9B1380000000, float 0x3FD5C53B80000000, float 0x3FD5C53B00000000, float 0x3FC6326AC0000000, float 0x3FC63269E0000000, float 0xBEE05CEB00000000, float 0xBEE086A320000000, float 0xBFC63269E0000000, float 0xBFC6326AC0000000, float 0xBFD5C53B80000000, float 0xBFD5C53B80000000, float 0xBFDF9B13E0000000, float 0xBFDF9B1460000000, float 0xBFE41CFE80000000, float 0x3FE7A693C0000000, float 0x3FEA477C20000000, float 0x3FEBE5DC40000000, float 0x3FEC71C6E0000000, float 0x3FEBE5DC40000000, float 0x3FEA477C20000000, float 0x3FE7A693C0000000, float 0xBFE41CFE80000000>, i32 %idx
  %v2 = extractelement <81 x float> <float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float 0xBFE41CFEA0000000, float 0xBFDF9B13E0000000, float 0xBFD5C53B80000000, float 0xBFC6326AC0000000, float 0x3EE0789320000000, float 0x3FC6326AC0000000, float 0x3FD5C53B80000000, float 0x3FDF9B13E0000000, float 0x3FE41CFEA0000000, float 0xBFE7A693C0000000, float 0x3FE7A693C0000000, float 0xBFEA477C20000000, float 0x3FEA477C20000000, float 0xBFEBE5DC40000000, float 0x3FEBE5DC40000000, float 0xBFEC71C720000000, float 0x3FEC71C6E0000000, float 0xBFEBE5DC60000000, float 0x3FEBE5DC40000000, float 0xBFEA477C20000000, float 0x3FEA477C20000000, float 0xBFE7A693C0000000, float 0x3FE7A69380000000, float 0xBFE41CFEA0000000, float 0xBFDF9B13E0000000, float 0xBFD5C53B80000000, float 0xBFC6326AC0000000, float 0x3EE0789320000000, float 0x3FC6326AC0000000, float 0x3FD5C53B80000000, float 0x3FDF9B13E0000000, float 0x3FE41CFE80000000>, i32 %idx
  %f = fadd float %v1, %v2
  %r1 = insertvalue <{i32, i32, i32, float}> undef, i32 %swo, 2
  %r2 = insertvalue <{i32, i32, i32, float}> %r1, float %f, 3
  ret <{i32, i32, i32, float}> %r2
}

; GCN-LABEL: {{^}}gs_ir_uses_scratch_offset:
; GFX9-FLATSCR: s_add_u32 flat_scratch_lo, s0, s5
; GFX9-FLATSCR: s_addc_u32 flat_scratch_hi, s1, 0

; GFX10-FLATSCR: s_add_u32 s0, s0, s5
; GFX10-FLATSCR: s_addc_u32 s1, s1, 0
; GFX10-FLATSCR: s_setreg_b32 hwreg(HW_REG_FLAT_SCR_LO), s0
; GFX10-FLATSCR: s_setreg_b32 hwreg(HW_REG_FLAT_SCR_HI), s1

; GFX9-FLATSCR-PAL-DAG: s_getpc_b64 s[0:1]
; GFX9-FLATSCR-PAL-DAG: s_mov_b32 s0, s8
; GFX9-FLATSCR-PAL-DAG: s_load_dwordx2 s[0:1], s[0:1], 0x0
; GFX9-FLATSCR-PAL-DAG: v_lshlrev_b32_e32 v0, 2, v0
; GFX9-FLATSCR-PAL-DAG: v_mov_b32_e32 v0, 0xbf20e7f4
; GFX9-FLATSCR-PAL-DAG: s_mov_b32 s0, 0
; GFX9-FLATSCR-PAL-DAG: s_waitcnt lgkmcnt(0)
; GFX9-FLATSCR-PAL-DAG: s_and_b32 s1, s1, 0xffff
; GFX9-FLATSCR-PAL-DAG: s_add_u32 flat_scratch_lo, s0, s5
; GFX9-FLATSCR-PAL-DAG: s_addc_u32 flat_scratch_hi, s1, 0

; GFX10-FLATSCR-PAL: s_getpc_b64 s[0:1]
; GFX10-FLATSCR-PAL: s_mov_b32 s0, s8
; GFX10-FLATSCR-PAL: s_load_dwordx2 s[0:1], s[0:1], 0x0
; GFX10-FLATSCR-PAL: s_waitcnt lgkmcnt(0)
; GFX10-FLATSCR-PAL: s_and_b32 s1, s1, 0xffff
; GFX10-FLATSCR-PAL: s_add_u32 s0, s0, s5
; GFX10-FLATSCR-PAL: s_addc_u32 s1, s1, 0
; GFX10-FLATSCR-PAL: s_setreg_b32 hwreg(HW_REG_FLAT_SCR_LO), s0
; GFX10-FLATSCR-PAL: s_setreg_b32 hwreg(HW_REG_FLAT_SCR_HI), s1

; MUBUF: s_mov_b32 s8, SCRATCH_RSRC_DWORD0
; FLATSCR-NOT: SCRATCH_RSRC_DWORD

; SIVI: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 offen
; SIVI: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 offen

; GFX9_10-MUBUF-DAG: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 offen
; GFX9_10-MUBUF-DAG: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 offen

; MUBUF-DAG: s_mov_b32 s2, s5

; FLATSCR-DAG: scratch_load_dword {{v[0-9]+}}, {{v[0-9]+}}, off
; FLATSCR-DAG: scratch_load_dword {{v[0-9]+}}, {{v[0-9]+}}, off

; GFX11-FLATSCR: scratch_load_b32 {{v[0-9]+}}, {{v[0-9]+}}, off
; GFX11-FLATSCR: scratch_load_b32 {{v[0-9]+}}, {{v[0-9]+}}, off
define amdgpu_gs <{i32, i32, i32, float}> @gs_ir_uses_scratch_offset(i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg %swo, i32 %idx) {
  %v1 = extractelement <81 x float> <float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float 0x3FE41CFEA0000000, float 0xBFE7A693C0000000, float 0xBFEA477C60000000, float 0xBFEBE5DC60000000, float 0xBFEC71C720000000, float 0xBFEBE5DC60000000, float 0xBFEA477C60000000, float 0xBFE7A693C0000000, float 0xBFE41CFEA0000000, float 0x3FDF9B13E0000000, float 0x3FDF9B1380000000, float 0x3FD5C53B80000000, float 0x3FD5C53B00000000, float 0x3FC6326AC0000000, float 0x3FC63269E0000000, float 0xBEE05CEB00000000, float 0xBEE086A320000000, float 0xBFC63269E0000000, float 0xBFC6326AC0000000, float 0xBFD5C53B80000000, float 0xBFD5C53B80000000, float 0xBFDF9B13E0000000, float 0xBFDF9B1460000000, float 0xBFE41CFE80000000, float 0x3FE7A693C0000000, float 0x3FEA477C20000000, float 0x3FEBE5DC40000000, float 0x3FEC71C6E0000000, float 0x3FEBE5DC40000000, float 0x3FEA477C20000000, float 0x3FE7A693C0000000, float 0xBFE41CFE80000000>, i32 %idx
  %v2 = extractelement <81 x float> <float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float 0xBFE41CFEA0000000, float 0xBFDF9B13E0000000, float 0xBFD5C53B80000000, float 0xBFC6326AC0000000, float 0x3EE0789320000000, float 0x3FC6326AC0000000, float 0x3FD5C53B80000000, float 0x3FDF9B13E0000000, float 0x3FE41CFEA0000000, float 0xBFE7A693C0000000, float 0x3FE7A693C0000000, float 0xBFEA477C20000000, float 0x3FEA477C20000000, float 0xBFEBE5DC40000000, float 0x3FEBE5DC40000000, float 0xBFEC71C720000000, float 0x3FEC71C6E0000000, float 0xBFEBE5DC60000000, float 0x3FEBE5DC40000000, float 0xBFEA477C20000000, float 0x3FEA477C20000000, float 0xBFE7A693C0000000, float 0x3FE7A69380000000, float 0xBFE41CFEA0000000, float 0xBFDF9B13E0000000, float 0xBFD5C53B80000000, float 0xBFC6326AC0000000, float 0x3EE0789320000000, float 0x3FC6326AC0000000, float 0x3FD5C53B80000000, float 0x3FDF9B13E0000000, float 0x3FE41CFE80000000>, i32 %idx
  %f = fadd float %v1, %v2
  %r1 = insertvalue <{i32, i32, i32, float}> undef, i32 %swo, 2
  %r2 = insertvalue <{i32, i32, i32, float}> %r1, float %f, 3
  ret <{i32, i32, i32, float}> %r2
}
