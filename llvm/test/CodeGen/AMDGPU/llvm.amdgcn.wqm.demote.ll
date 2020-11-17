; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SI,GCN-64,PRE-GFX10 %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX9,GCN-64,PRE-GFX10 %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -mattr=+wavefrontsize32,-wavefrontsize64 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX10,GCN-32 %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -mattr=-wavefrontsize32,+wavefrontsize64 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX10,GCN-64 %s

; GCN-LABEL: {{^}}static_exact:
; GCN-32: v_cmp_gt_f32_e32 [[CMP:vcc_lo]], 0, v0
; GCN-64: v_cmp_gt_f32_e32 [[CMP:vcc]], 0, v0
; GCN-32: s_mov_b32 exec_lo, 0
; GCN-64: s_mov_b64 exec, 0
; GCN: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1.0, [[CMP]]
; GCN: exp mrt1 v0, v0, v0, v0 done vm
define amdgpu_ps void @static_exact(float %arg0, float %arg1) {
.entry:
  %c0 = fcmp olt float %arg0, 0.000000e+00
  %c1 = fcmp oge float %arg1, 0.0
  call void @llvm.amdgcn.wqm.demote(i1 false)
  %tmp1 = select i1 %c0, float 1.000000e+00, float 0.000000e+00
  call void @llvm.amdgcn.exp.f32(i32 1, i32 15, float %tmp1, float %tmp1, float %tmp1, float %tmp1, i1 true, i1 true) #0
  ret void
}

; GCN-LABEL: {{^}}dynamic_exact:
; GCN-32: v_cmp_le_f32_e64 [[CND:s[0-9]+]], 0, v1
; GCN-64: v_cmp_le_f32_e64 [[CND:s\[[0-9]+:[0-9]+\]]], 0, v1
; GCN-32: v_cmp_gt_f32_e32 [[CMP:vcc_lo]], 0, v0
; GCN-64: v_cmp_gt_f32_e32 [[CMP:vcc]], 0, v0
; GCN-32: s_and_b32 exec_lo, exec_lo, [[CND]]
; GCN-64: s_and_b64 exec, exec, [[CND]]
; GCN: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1.0, [[CMP]]
; GCN: exp mrt1 v0, v0, v0, v0 done vm
define amdgpu_ps void @dynamic_exact(float %arg0, float %arg1) {
.entry:
  %c0 = fcmp olt float %arg0, 0.000000e+00
  %c1 = fcmp oge float %arg1, 0.0
  call void @llvm.amdgcn.wqm.demote(i1 %c1)
  %tmp1 = select i1 %c0, float 1.000000e+00, float 0.000000e+00
  call void @llvm.amdgcn.exp.f32(i32 1, i32 15, float %tmp1, float %tmp1, float %tmp1, float %tmp1, i1 true, i1 true) #0
  ret void
}

; GCN-LABEL: {{^}}branch:
; GCN-32: s_and_saveexec_b32 s1, s0
; GCN-64: s_and_saveexec_b64 s[2:3], s[0:1]
; GCN-32: s_xor_b32 s0, exec_lo, s1
; GCN-64: s_xor_b64 s[0:1], exec, s[2:3]
; GCN-32: s_mov_b32 exec_lo, 0
; GCN-64: s_mov_b64 exec, 0
; GCN-32: s_or_b32 exec_lo, exec_lo, s0
; GCN-64: s_or_b64 exec, exec, s[0:1]
; GCN: v_cndmask_b32_e64 v0, 0, 1.0, vcc
; GCN: exp mrt1 v0, v0, v0, v0 done vm
define amdgpu_ps void @branch(float %arg0, float %arg1) {
.entry:
  %i0 = fptosi float %arg0 to i32
  %i1 = fptosi float %arg1 to i32
  %c0 = or i32 %i0, %i1
  %c1 = and i32 %c0, 1
  %c2 = icmp eq i32 %c1, 0
  br i1 %c2, label %.continue, label %.demote

.demote:
  call void @llvm.amdgcn.wqm.demote(i1 false)
  br label %.continue

.continue:
  %tmp1 = select i1 %c2, float 1.000000e+00, float 0.000000e+00
  call void @llvm.amdgcn.exp.f32(i32 1, i32 15, float %tmp1, float %tmp1, float %tmp1, float %tmp1, i1 true, i1 true) #0
  ret void
}


; GCN-LABEL: {{^}}wqm_demote_1:
; GCN-NEXT: ; %.entry
; GCN-32: s_mov_b32 [[ORIG:s[0-9]+]], exec_lo
; GCN-64: s_mov_b64 [[ORIG:s\[[0-9]+:[0-9]+\]]], exec
; GCN-32: s_wqm_b32 exec_lo, exec_lo
; GCN-64: s_wqm_b64 exec, exec
; GCN: ; %.demote
; GCN-32-NEXT: s_andn2_b32 [[LIVE:s[0-9]+]], [[ORIG]], exec_lo
; GCN-64-NEXT: s_andn2_b64 [[LIVE:s\[[0-9]+:[0-9]+\]]], [[ORIG]], exec
; GCN: s_cbranch_scc0 [[EARLYTERM:BB[0-9]+_[0-9]+]]
; GCN: ; %.demote
; GCN-32-NEXT: s_wqm_b32 [[LIVEWQM0:s[0-9]+]], [[LIVE]]
; GCN-64-NEXT: s_wqm_b64 [[LIVEWQM0:s\[[0-9]+:[0-9]+\]]], [[LIVE]]
; GCN-32-NEXT: s_and_b32 exec_lo, exec_lo, [[LIVEWQM0]]
; GCN-64-NEXT: s_and_b64 exec, exec, [[LIVEWQM0]]
; GCN: ; %.continue
; GCN-32: s_wqm_b32 [[LIVEWQM1:s[0-9]+]], [[LIVE]]
; GCN-64: s_wqm_b64 [[LIVEWQM1:s\[[0-9]+:[0-9]+\]]], [[LIVE]]
; GCN-32-NEXT: s_and_b32 exec_lo, exec_lo, [[LIVEWQM1]]
; GCN-64-NEXT: s_and_b64 exec, exec, [[LIVEWQM1]]
; GCN: s_cbranch_execz
; GCN: v_add_f32_e32
; GCN-32: s_and_b32 exec_lo, exec_lo, [[LIVE]]
; GCN-64: s_and_b64 exec, exec, [[LIVE]]
; GCN: image_sample
; GCN: [[EARLYTERM]]:
; GCN-NEXT: s_mov
; GCN-NEXT: exp null off, off, off, off done vm
; GCN-NEXT: s_endpgm
define amdgpu_ps <4 x float> @wqm_demote_1(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, i32 %idx, float %data, float %coord, float %coord2, float %z) {
.entry:
  %z.cmp = fcmp olt float %z, 0.0
  br i1 %z.cmp, label %.continue, label %.demote

.demote:
  call void @llvm.amdgcn.wqm.demote(i1 false)
  br label %.continue

.continue:
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %coord, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0
  %tex0 = extractelement <4 x float> %tex, i32 0
  %tex1 = extractelement <4 x float> %tex, i32 0
  %coord1 = fadd float %tex0, %tex1
  %rtex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %coord1, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0

  ret <4 x float> %rtex
}

; GCN-LABEL: {{^}}wqm_demote_2:
; GCN-NEXT: ; %.entry
; GCN-32: s_mov_b32 [[ORIG:s[0-9]+]], exec_lo
; GCN-64: s_mov_b64 [[ORIG:s\[[0-9]+:[0-9]+\]]], exec
; GCN-32: s_wqm_b32 exec_lo, exec_lo
; GCN-64: s_wqm_b64 exec, exec
; GCN: image_sample
; GCN: ; %.demote
; GCN-32-NEXT: s_andn2_b32 [[LIVE:s[0-9]+]], [[ORIG]], exec
; GCN-64-NEXT: s_andn2_b64 [[LIVE:s\[[0-9]+:[0-9]+\]]], [[ORIG]], exec
; GCN: s_cbranch_scc0 [[EARLYTERM:BB[0-9]+_[0-9]+]]
; GCN: ; %.demote
; GCN-32-NEXT: s_wqm_b32 [[LIVEWQM0:s[0-9]+]], [[LIVE]]
; GCN-64-NEXT: s_wqm_b64 [[LIVEWQM0:s\[[0-9]+:[0-9]+\]]], [[LIVE]]
; GCN-32-NEXT: s_and_b32 exec_lo, exec_lo, [[LIVEWQM0]]
; GCN-64-NEXT: s_and_b64 exec, exec, [[LIVEWQM0]]
; GCN: ; %.continue
; GCN-32: s_wqm_b32 [[LIVEWQM1:s[0-9]+]], [[LIVE]]
; GCN-64: s_wqm_b64 [[LIVEWQM1:s\[[0-9]+:[0-9]+\]]], [[LIVE]]
; GCN-32-NEXT: s_and_b32 exec_lo, exec_lo, [[LIVEWQM1]]
; GCN-64-NEXT: s_and_b64 exec, exec, [[LIVEWQM1]]
; GCN: s_cbranch_execz
; GCN: v_add_f32_e32
; GCN-32: s_and_b32 exec_lo, exec_lo, [[LIVE]]
; GCN-64: s_and_b64 exec, exec, [[LIVE]]
; GCN: image_sample
; GCN: [[EARLYTERM]]:
; GCN-NEXT: s_mov
; GCN-NEXT: exp null off, off, off, off done vm
; GCN-NEXT: s_endpgm
define amdgpu_ps <4 x float> @wqm_demote_2(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, i32 %idx, float %data, float %coord, float %coord2, float %z) {
.entry:
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %coord, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0
  %tex0 = extractelement <4 x float> %tex, i32 0
  %tex1 = extractelement <4 x float> %tex, i32 0
  %z.cmp = fcmp olt float %tex0, 0.0
  br i1 %z.cmp, label %.continue, label %.demote

.demote:
  call void @llvm.amdgcn.wqm.demote(i1 false)
  br label %.continue

.continue:
  %coord1 = fadd float %tex0, %tex1
  %rtex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %coord1, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0

  ret <4 x float> %rtex
}

; GCN-LABEL: {{^}}wqm_demote_dynamic:
; GCN-NEXT: ; %.entry
; GCN-32: s_mov_b32 [[ORIG:s[0-9]+]], exec_lo
; GCN-64: s_mov_b64 [[ORIG:s\[[0-9]+:[0-9]+\]]], exec
; GCN-32: s_wqm_b32 exec_lo, exec_lo
; GCN-64: s_wqm_b64 exec, exec
; GCN: image_sample
; GCN: v_cmp_gt_f32_e32 vcc
; GCN-32-NEXT: s_xor_b32 [[TMP:s[0-9]+]], vcc_lo, exec
; GCN-64-NEXT: s_xor_b64 [[TMP:s\[[0-9]+:[0-9]+\]]], vcc, exec
; GCN-32-NEXT: s_andn2_b32 [[LIVE:s[0-9]+]], [[ORIG]], [[TMP]]
; GCN-64-NEXT: s_andn2_b64 [[LIVE:s\[[0-9]+:[0-9]+\]]], [[ORIG]], [[TMP]]
; GCN-NEXT: s_cbranch_scc0 [[EARLYTERM:BB[0-9]+_[0-9]+]]
; GCN-32: s_wqm_b32 [[LIVEWQM0:s[0-9]+]], [[LIVE]]
; GCN-64: s_wqm_b64 [[LIVEWQM0:s\[[0-9]+:[0-9]+\]]], [[LIVE]]
; GCN-32-NEXT: s_and_b32 exec_lo, exec_lo, [[LIVEWQM0]]
; GCN-64-NEXT: s_and_b64 exec, exec, [[LIVEWQM0]]
; GCN: s_cbranch_execz
; GCN: v_add_f32_e32
; GCN-32: s_and_b32 exec_lo, exec_lo, [[LIVE]]
; GCN-64: s_and_b64 exec, exec, [[LIVE]]
; GCN: image_sample
; GCN: [[EARLYTERM]]:
; GCN-NEXT: s_mov
; GCN-NEXT: exp null off, off, off, off done vm
; GCN-NEXT: s_endpgm
define amdgpu_ps <4 x float> @wqm_demote_dynamic(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, i32 %idx, float %data, float %coord, float %coord2, float %z) {
.entry:
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %coord, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0
  %tex0 = extractelement <4 x float> %tex, i32 0
  %tex1 = extractelement <4 x float> %tex, i32 0
  %z.cmp = fcmp olt float %tex0, 0.0
  call void @llvm.amdgcn.wqm.demote(i1 %z.cmp)
  %coord1 = fadd float %tex0, %tex1
  %rtex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %coord1, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0

  ret <4 x float> %rtex
}


; GCN-LABEL: {{^}}wqm_deriv:
; GCN-NEXT: ; %.entry
; GCN-32: s_mov_b32 [[ORIG:s[0-9]+]], exec_lo
; GCN-64: s_mov_b64 [[ORIG:s\[[0-9]+:[0-9]+\]]], exec
; GCN-32: s_wqm_b32 exec_lo, exec_lo
; GCN-64: s_wqm_b64 exec, exec
; GCN: ; %.demote0
; GCN-32-NEXT: s_andn2_b32 [[LIVE:s[0-9]+]], [[ORIG]], exec
; GCN-64-NEXT: s_andn2_b64 [[LIVE:s\[[0-9]+:[0-9]+\]]], [[ORIG]], exec
; GCN-NEXT: s_cbranch_scc0 [[EARLYTERM:BB[0-9]+_[0-9]+]]
; GCN-32: s_wqm_b32 [[LIVEWQM0:s[0-9]+]], [[LIVE]]
; GCN-64: s_wqm_b64 [[LIVEWQM0:s\[[0-9]+:[0-9]+\]]], [[LIVE]]
; GCN-32-NEXT: s_and_b32 exec_lo, exec_lo, [[LIVEWQM0]]
; GCN-64-NEXT: s_and_b64 exec, exec, [[LIVEWQM0]]
; GCN-NOT: s_cbranch_execz
; GCN: ; %.continue0
; GCN-32: s_wqm_b32 [[LIVEWQM1:s[0-9]+]], [[LIVE]]
; GCN-64: s_wqm_b64 [[LIVEWQM1:s\[[0-9]+:[0-9]+\]]], [[LIVE]]
; GCN-32-NEXT: s_and_b32 exec_lo, exec_lo, [[LIVEWQM1]]
; GCN-64-NEXT: s_and_b64 exec, exec, [[LIVEWQM1]]
; GCN: s_cbranch_execz
; GCN: v_cndmask_b32_e64 [[DST:v[0-9]+]], 1.0, 0, [[LIVE]]
; GCN-32: s_and_b32 exec_lo, exec_lo, [[LIVE]]
; GCN-64: s_and_b64 exec, exec, [[LIVE]]
; GCN: ; %.demote1
; GCN-32-NEXT: s_andn2_b32 [[LIVE]], [[LIVE]], exec
; GCN-64-NEXT: s_andn2_b64 [[LIVE]], [[LIVE]], exec
; GCN-NEXT: s_cbranch_scc0 [[EARLYTERM]]
; GCN-NEXT: ; %.demote1
; GCN-32-NEXT: s_mov_b32 exec_lo, 0
; GCN-64-NEXT: s_mov_b64 exec, 0
; GCN: ; %.continue1
; GCN: exp mrt0
; GCN: [[EARLYTERM]]:
; GCN-NEXT: s_mov
; GCN-NEXT: exp null off, off, off, off done vm
; GCN-NEXT: s_endpgm
define amdgpu_ps void @wqm_deriv(<2 x float> %input, float %arg, i32 %index) {
.entry:
  %p0 = extractelement <2 x float> %input, i32 0
  %p1 = extractelement <2 x float> %input, i32 1
  %x0 = call float @llvm.amdgcn.interp.p1(float %p0, i32 immarg 0, i32 immarg 0, i32 %index) #2
  %x1 = call float @llvm.amdgcn.interp.p2(float %x0, float %p1, i32 immarg 0, i32 immarg 0, i32 %index) #2
  %argi = fptosi float %arg to i32
  %cond0 = icmp eq i32 %argi, 0
  br i1 %cond0, label %.continue0, label %.demote0

.demote0:
  call void @llvm.amdgcn.wqm.demote(i1 false)
  br label %.continue0

.continue0:
  %live = call i1 @llvm.amdgcn.wqm.helper()
  %live.cond = select i1 %live, i32 0, i32 1065353216
  %live.v0 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %live.cond, i32 85, i32 15, i32 15, i1 true)
  %live.v0f = bitcast i32 %live.v0 to float
  %live.v1 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %live.cond, i32 0, i32 15, i32 15, i1 true)
  %live.v1f = bitcast i32 %live.v1 to float
  %v0 = fsub float %live.v0f, %live.v1f
  %v0.wqm = call float @llvm.amdgcn.wqm.f32(float %v0)
  %cond1 = fcmp oeq float %v0.wqm, 0.000000e+00
  %cond2 = and i1 %live, %cond1
  br i1 %cond2, label %.continue1, label %.demote1

.demote1:
  call void @llvm.amdgcn.wqm.demote(i1 false)
  br label %.continue1

.continue1:
  call void @llvm.amdgcn.exp.compr.v2f16(i32 immarg 0, i32 immarg 15, <2 x half> <half 0xH3C00, half 0xH0000>, <2 x half> <half 0xH0000, half 0xH3C00>, i1 immarg true, i1 immarg true) #3
  ret void
}

; GCN-LABEL: {{^}}wqm_deriv_loop:
; GCN-NEXT: ; %.entry
; GCN-32: s_mov_b32 [[ORIG:s[0-9]+]], exec_lo
; GCN-64: s_mov_b64 [[ORIG:s\[[0-9]+:[0-9]+\]]], exec
; GCN-32: s_wqm_b32 exec_lo, exec_lo
; GCN-64: s_wqm_b64 exec, exec
; GCN: ; %.demote0
; GCN-32-NEXT: s_andn2_b32 [[LIVE:s[0-9]+]], [[ORIG]], exec
; GCN-64-NEXT: s_andn2_b64 [[LIVE:s\[[0-9]+:[0-9]+\]]], [[ORIG]], exec
; GCN-NEXT: s_cbranch_scc0 [[EARLYTERM:BB[0-9]+_[0-9]+]]
; GCN-32: s_wqm_b32 [[LIVEWQM0:s[0-9]+]], [[LIVE]]
; GCN-64: s_wqm_b64 [[LIVEWQM0:s\[[0-9]+:[0-9]+\]]], [[LIVE]]
; GCN-32-NEXT: s_and_b32 exec_lo, exec_lo, [[LIVEWQM0]]
; GCN-64-NEXT: s_and_b64 exec, exec, [[LIVEWQM0]]
; GCN-NOT: s_cbranch_execz
; GCN: ; %.continue0.preheader
; GCN-32: s_wqm_b32 [[LIVEWQM1:s[0-9]+]], [[LIVE]]
; GCN-64: s_wqm_b64 [[LIVEWQM1:s\[[0-9]+:[0-9]+\]]], [[LIVE]]
; GCN-32-NEXT: s_and_b32 exec_lo, exec_lo, [[LIVEWQM1]]
; GCN-64-NEXT: s_and_b64 exec, exec, [[LIVEWQM1]]
; GCN: s_cbranch_execz
; GCN: ; %.demote1
; GCN-32: s_andn2_b32 [[LIVE]], [[LIVE]], exec
; GCN-64: s_andn2_b64 [[LIVE]], [[LIVE]], exec
; GCN-NEXT: s_cbranch_scc0 [[EARLYTERM]]
; GCN-NOT: s_cbranch_execz
; GCN: ; %.continue1
; GCN-32: s_or_b32 exec_lo
; GCN-64: s_or_b64 exec
; GCN: ; %.continue0
; PRE-GFX10: v_cndmask_b32_e64 [[DST:v[0-9]+]], [[SRC:v[0-9]+]], 0, [[LIVE]]
; GFX10: v_cndmask_b32_e64 [[DST:v[0-9]+]], [[SRC:s[0-9]+]], 0, [[LIVE]]
; GCN: ; %.return
; GCN-32: s_and_b32 exec_lo, exec_lo, [[LIVE]]
; GCN-64: s_and_b64 exec, exec, [[LIVE]]
; GCN: exp mrt0
; GCN: [[EARLYTERM]]:
; GCN-NEXT: s_mov
; GCN-NEXT: exp null off, off, off, off done vm
; GCN-NEXT: s_endpgm
define amdgpu_ps void @wqm_deriv_loop(<2 x float> %input, float %arg, i32 %index, i32 %limit) {
.entry:
  %p0 = extractelement <2 x float> %input, i32 0
  %p1 = extractelement <2 x float> %input, i32 1
  %x0 = call float @llvm.amdgcn.interp.p1(float %p0, i32 immarg 0, i32 immarg 0, i32 %index) #2
  %x1 = call float @llvm.amdgcn.interp.p2(float %x0, float %p1, i32 immarg 0, i32 immarg 0, i32 %index) #2
  %argi = fptosi float %arg to i32
  %cond0 = icmp eq i32 %argi, 0
  br i1 %cond0, label %.continue0, label %.demote0

.demote0:
  call void @llvm.amdgcn.wqm.demote(i1 false)
  br label %.continue0

.continue0:
  %count = phi i32 [ 0, %.entry ], [ 0, %.demote0 ], [ %next, %.continue1 ]
  %live = call i1 @llvm.amdgcn.wqm.helper()
  %live.cond = select i1 %live, i32 0, i32 %count
  %live.v0 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %live.cond, i32 85, i32 15, i32 15, i1 true)
  %live.v0f = bitcast i32 %live.v0 to float
  %live.v1 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %live.cond, i32 0, i32 15, i32 15, i1 true)
  %live.v1f = bitcast i32 %live.v1 to float
  %v0 = fsub float %live.v0f, %live.v1f
  %v0.wqm = call float @llvm.amdgcn.wqm.f32(float %v0)
  %cond1 = fcmp oeq float %v0.wqm, 0.000000e+00
  %cond2 = and i1 %live, %cond1
  br i1 %cond2, label %.continue1, label %.demote1

.demote1:
  call void @llvm.amdgcn.wqm.demote(i1 false)
  br label %.continue1

.continue1:
  %next = add i32 %count, 1
  %loop.cond = icmp slt i32 %next, %limit
  br i1 %loop.cond, label %.continue0, label %.return

.return:
  call void @llvm.amdgcn.exp.compr.v2f16(i32 immarg 0, i32 immarg 15, <2 x half> <half 0xH3C00, half 0xH0000>, <2 x half> <half 0xH0000, half 0xH3C00>, i1 immarg true, i1 immarg true) #3
  ret void
}

declare void @llvm.amdgcn.wqm.demote(i1) #0
declare i1 @llvm.amdgcn.wqm.helper() #0
declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare float @llvm.amdgcn.wqm.f32(float) #1
declare float @llvm.amdgcn.interp.p1(float, i32 immarg, i32 immarg, i32) #2
declare float @llvm.amdgcn.interp.p2(float, float, i32 immarg, i32 immarg, i32) #2
declare void @llvm.amdgcn.exp.compr.v2f16(i32 immarg, i32 immarg, <2 x half>, <2 x half>, i1 immarg, i1 immarg) #3
declare i32 @llvm.amdgcn.mov.dpp.i32(i32, i32 immarg, i32 immarg, i32 immarg, i1 immarg) #4

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind readnone speculatable }
attributes #3 = { inaccessiblememonly nounwind }
attributes #4 = { convergent nounwind readnone }
