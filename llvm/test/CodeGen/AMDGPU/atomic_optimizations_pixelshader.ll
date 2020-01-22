; RUN: llc -mtriple=amdgcn-- -amdgpu-atomic-optimizations=true -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN64,GFX7LESS %s
; RUN: llc  -mtriple=amdgcn-- -mcpu=tonga -mattr=-flat-for-global -amdgpu-atomic-optimizations=true -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN64,GFX8MORE,GFX8MORE64,DPPCOMB %s
; RUN: llc -mtriple=amdgcn-- -mcpu=gfx900 -mattr=-flat-for-global -amdgpu-atomic-optimizations=true -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN64,GFX8MORE,GFX8MORE64,DPPCOMB %s
; RUN: llc -mtriple=amdgcn-- -mcpu=gfx1010 -mattr=-wavefrontsize32,+wavefrontsize64 -mattr=-flat-for-global -amdgpu-atomic-optimizations=true -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN64,GFX8MORE,GFX8MORE64 %s
; RUN: llc -mtriple=amdgcn-- -mcpu=gfx1010 -mattr=+wavefrontsize32,-wavefrontsize64 -mattr=-flat-for-global -amdgpu-atomic-optimizations=true -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN32,GFX8MORE,GFX8MORE32 %s

declare i1 @llvm.amdgcn.wqm.vote(i1)
declare i32 @llvm.amdgcn.raw.buffer.atomic.add(i32, <4 x i32>, i32, i32, i32 immarg)
declare void @llvm.amdgcn.raw.buffer.store.f32(float, <4 x i32>, i32, i32, i32 immarg)

; Show that what the atomic optimization pass will do for raw buffers.

; GCN-LABEL: add_i32_constant:
; %bb.{{[0-9]+}}:
; GCN32: v_cmp_ne_u32_e64 s[[exec_lo:[0-9]+]], 1, 0
; GCN64: v_cmp_ne_u32_e64 s{{\[}}[[exec_lo:[0-9]+]]:[[exec_hi:[0-9]+]]{{\]}}, 1, 0
; GCN: v_mbcnt_lo_u32_b32{{(_e[0-9]+)?}} v[[mbcnt:[0-9]+]], s[[exec_lo]], 0
; GCN64: v_mbcnt_hi_u32_b32{{(_e[0-9]+)?}} v[[mbcnt]], s[[exec_hi]], v[[mbcnt]]
; GCN: v_cmp_eq_u32{{(_e[0-9]+)?}} vcc{{(_lo)?}}, 0, v[[mbcnt]]
; GCN32: s_bcnt1_i32_b32 s[[popcount:[0-9]+]], s[[exec_lo]]
; GCN64: s_bcnt1_i32_b64 s[[popcount:[0-9]+]], s{{\[}}[[exec_lo]]:[[exec_hi]]{{\]}}
; GCN: v_mul_u32_u24{{(_e[0-9]+)?}} v[[value:[0-9]+]], s[[popcount]], 5
; GCN: buffer_atomic_add v[[value]]
; GCN: v_readfirstlane_b32 s{{[0-9]+}}, v[[value]]
define amdgpu_ps void @add_i32_constant(<4 x i32> inreg %out, <4 x i32> inreg %inout) {
entry:
  %cond1 = call i1 @llvm.amdgcn.wqm.vote(i1 true)
  %old = call i32 @llvm.amdgcn.raw.buffer.atomic.add(i32 5, <4 x i32> %inout, i32 0, i32 0, i32 0)
  %cond2 = call i1 @llvm.amdgcn.wqm.vote(i1 true)
  %cond = and i1 %cond1, %cond2
  br i1 %cond, label %if, label %else
if:
  %bitcast = bitcast i32 %old to float
  call void @llvm.amdgcn.raw.buffer.store.f32(float %bitcast, <4 x i32> %out, i32 0, i32 0, i32 0)
  ret void
else:
  ret void
}

; GCN-LABEL: add_i32_varying:
; GFX7LESS-NOT: v_mbcnt_lo_u32_b32
; GFX7LESS-NOT: v_mbcnt_hi_u32_b32
; GFX8MORE32: v_cmp_ne_u32_e64 s[[exec_lo:[0-9]+]], 1, 0
; GFX8MORE64: v_cmp_ne_u32_e64 s{{\[}}[[exec_lo:[0-9]+]]:[[exec_hi:[0-9]+]]{{\]}}, 1, 0
; GFX8MORE: v_mbcnt_lo_u32_b32{{(_e[0-9]+)?}} v[[mbcnt:[0-9]+]], s[[exec_lo]], 0
; GFX8MORE64: v_mbcnt_hi_u32_b32{{(_e[0-9]+)?}} v[[mbcnt]], s[[exec_hi]], v[[mbcnt]]
; DPPCOMB: v_add_u32_dpp
; DPPCOMB: v_add_u32_dpp
; GFX8MORE32: v_readlane_b32 s[[scalar_value:[0-9]+]], v{{[0-9]+}}, 31
; GFX8MORE64: v_readlane_b32 s[[scalar_value:[0-9]+]], v{{[0-9]+}}, 63
; GFX8MORE: v_cmp_eq_u32{{(_e[0-9]+)?}} vcc{{(_lo)?}}, 0, v[[mbcnt]]
; GFX8MORE: v_mov_b32{{(_e[0-9]+)?}} v[[value:[0-9]+]], s[[scalar_value]]
; GFX8MORE: buffer_atomic_add v[[value]]
; GFX8MORE: v_readfirstlane_b32 s{{[0-9]+}}, v[[value]]
define amdgpu_ps void @add_i32_varying(<4 x i32> inreg %out, <4 x i32> inreg %inout, i32 %val) {
entry:
  %cond1 = call i1 @llvm.amdgcn.wqm.vote(i1 true)
  %old = call i32 @llvm.amdgcn.raw.buffer.atomic.add(i32 %val, <4 x i32> %inout, i32 0, i32 0, i32 0)
  %cond2 = call i1 @llvm.amdgcn.wqm.vote(i1 true)
  %cond = and i1 %cond1, %cond2
  br i1 %cond, label %if, label %else
if:
  %bitcast = bitcast i32 %old to float
  call void @llvm.amdgcn.raw.buffer.store.f32(float %bitcast, <4 x i32> %out, i32 0, i32 0, i32 0)
  ret void
else:
  ret void
}
