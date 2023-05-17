; RUN: llc -global-isel=0 -march=amdgcn -mcpu=gfx1010 -mattr=+wavefrontsize32,-wavefrontsize64 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX1032 %s
; RUN: llc -global-isel=1 -march=amdgcn -mcpu=gfx1010 -mattr=+wavefrontsize32,-wavefrontsize64 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX1032 %s
; RUN: llc -global-isel=0 -march=amdgcn -mcpu=gfx1010 -mattr=-wavefrontsize32,+wavefrontsize64 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX1064 %s
; RUN: llc -global-isel=1 -march=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize32,-wavefrontsize64 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX1032 %s
; RUN: llc -global-isel=0 -march=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize32,-wavefrontsize64 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX1032 %s
; RUN: llc -global-isel=1 -march=amdgcn -mcpu=gfx1010 -mattr=-wavefrontsize32,+wavefrontsize64 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX1064 %s
; RUN: llc -global-isel=0 -march=amdgcn -mcpu=gfx1100 -mattr=-wavefrontsize32,+wavefrontsize64 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX1064 %s
; RUN: llc -global-isel=1 -march=amdgcn -mcpu=gfx1100 -mattr=-wavefrontsize32,+wavefrontsize64 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX1064 %s

; GCN-LABEL: {{^}}test_init_exec:
; GFX1032: s_mov_b32 exec_lo, 0x12345
; GFX1064: s_mov_b64 exec, 0x12345
; GCN: v_add_f32_e32 v0,
define amdgpu_ps float @test_init_exec(float %a, float %b) {
main_body:
  %s = fadd float %a, %b
  call void @llvm.amdgcn.init.exec(i64 74565)
  ret float %s
}

; GCN-LABEL: {{^}}test_init_exec_from_input:
; GCN: s_bfe_u32 s0, s3, 0x70008
; GFX1032: s_bfm_b32 exec_lo, s0, 0
; GFX1032: s_cmp_eq_u32 s0, 32
; GFX1032: s_cmov_b32 exec_lo, -1
; GFX1064: s_bfm_b64 exec, s0, 0
; GFX1064: s_cmp_eq_u32 s0, 64
; GFX1064: s_cmov_b64 exec, -1
; GCN: v_add_f32_e32 v0,
define amdgpu_ps float @test_init_exec_from_input(i32 inreg, i32 inreg, i32 inreg, i32 inreg %count, float %a, float %b) {
main_body:
  %s = fadd float %a, %b
  call void @llvm.amdgcn.init.exec.from.input(i32 %count, i32 8)
  ret float %s
}

declare void @llvm.amdgcn.init.exec(i64)
declare void @llvm.amdgcn.init.exec.from.input(i32, i32)
