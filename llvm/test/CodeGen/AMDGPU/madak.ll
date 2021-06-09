; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX6,GFX6_8_9,MAD %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX8,GFX6_8_9,GFX8_9,GFX8PLUS,MAD %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX9,GFX6_8_9,GFX8_9,GFX8PLUS,MAD %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX10PLUS,GFX8PLUS,GFX10-MAD %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs -fp-contract=fast < %s | FileCheck -check-prefixes=GCN,GFX10PLUS,GFX8PLUS,FMA %s
; RUN: llc -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX8PLUS,GFX11-MAD %s
; RUN: llc -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs -fp-contract=fast < %s | FileCheck -check-prefixes=GCN,GFX10PLUS,GFX8PLUS,FMA %s

declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
declare float @llvm.fabs.f32(float) nounwind readnone

; GCN-LABEL: {{^}}madak_f32:
; GFX6:      buffer_load_dword [[VA:v[0-9]+]]
; GFX6:      buffer_load_dword [[VB:v[0-9]+]]
; GFX8:      {{flat|global}}_load_dword [[VA:v[0-9]+]]
; GFX8:      {{flat|global}}_load_dword [[VB:v[0-9]+]]
; GFX9:      {{flat|global}}_load_dword [[VA:v[0-9]+]]
; GFX9:      {{flat|global}}_load_dword [[VB:v[0-9]+]]
; GFX10PLUS: {{flat|global}}_load_{{dword|b32}} [[VA:v[0-9]+]]
; GFX10PLUS: {{flat|global}}_load_{{dword|b32}} [[VB:v[0-9]+]]
; GFX11-MAD: global_load_b32 [[VA:v[0-9]+]]
; GFX11-MAD: global_load_b32 [[VB:v[0-9]+]]
; GFX11-MAD: v_mul_f32_e32 [[VADD:v[0-9]+]], [[VA]], [[VB]]
; GFX11-MAD: v_add_f32_e32 {{v[0-9]+}}, 0x41200000, [[VADD]]
; MAD:       v_madak_f32 {{v[0-9]+}}, [[VA]], [[VB]], 0x41200000
; GFX10-MAD: v_madak_f32 {{v[0-9]+}}, [[VA]], [[VB]], 0x41200000
; FMA:       v_fmaak_f32 {{v[0-9]+}}, [[VA]], [[VB]], 0x41200000
define amdgpu_kernel void @madak_f32(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in.a, float addrspace(1)* noalias %in.b) #0 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %in.a.gep = getelementptr float, float addrspace(1)* %in.a, i32 %tid
  %in.b.gep = getelementptr float, float addrspace(1)* %in.b, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid

  %a = load float, float addrspace(1)* %in.a.gep, align 4
  %b = load float, float addrspace(1)* %in.b.gep, align 4

  %mul = fmul float %a, %b
  %madak = fadd float %mul, 10.0
  store float %madak, float addrspace(1)* %out.gep, align 4
  ret void
}

; Make sure this is only folded with one use. This is a code size
; optimization and if we fold the immediate multiple times, we'll undo
; it.

; GCN-LABEL: {{^}}madak_2_use_f32:
; GFX9:         v_mov_b32_e32 [[VK:v[0-9]+]], 0x41200000
; GFX10PLUS:    v_mov_b32_e32 [[VK:v[0-9]+]], 0x41200000
; GFX11-MAD:    s_mov_b32 [[SK:s[0-9]+]], 0x41200000
; GFX6-DAG:     buffer_load_dword [[VA:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; GFX6-DAG:     buffer_load_dword [[VB:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; GFX6-DAG:     buffer_load_dword [[VC:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8
; GFX8PLUS:     {{flat|global}}_load_{{dword|b32}} [[VA:v[0-9]+]],
; GFX8PLUS:     {{flat|global}}_load_{{dword|b32}} [[VB:v[0-9]+]],
; GFX8PLUS:     {{flat|global}}_load_{{dword|b32}} [[VC:v[0-9]+]],
; GFX6-DAG:     v_mov_b32_e32 [[VK:v[0-9]+]], 0x41200000
; GFX8-DAG:     v_mov_b32_e32 [[VK:v[0-9]+]], 0x41200000
; GFX6_8_9-DAG: v_madak_f32 {{v[0-9]+}}, [[VA]], [[VB]], 0x41200000
; GFX10-MAD-DAG:v_madak_f32 {{v[0-9]+}}, [[VA]], [[VB]], 0x41200000
; FMA-DAG:      v_fmaak_f32 {{v[0-9]+}}, [[VA]], [[VB]], 0x41200000
; MAD-DAG:      v_mac_f32_e32 [[VK]], [[VA]], [[VC]]
; FMA-DAG:      v_fmac_f32_e32 [[VK]], [[VA]], [[VC]]
; GFX11-MAD:    v_mul_f32_e32 [[VMULAB:v[0-9]+]], [[VA]], [[VB]]
; GFX11-MAD:    v_mul_f32_e32 [[VMULAC:v[0-9]+]], [[VA]], [[VC]]
; GFX11-MAD:    v_dual_add_f32 {{v[0-9]+}}, [[SK]], [[VMULAC]] :: v_dual_add_f32 {{v[0-9]+}}, [[SK]], [[VMULAB]]
; GCN:          s_endpgm
define amdgpu_kernel void @madak_2_use_f32(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in) #0 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone

  %in.gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %in.gep.1 = getelementptr float, float addrspace(1)* %in.gep.0, i32 1
  %in.gep.2 = getelementptr float, float addrspace(1)* %in.gep.0, i32 2

  %out.gep.0 = getelementptr float, float addrspace(1)* %out, i32 %tid
  %out.gep.1 = getelementptr float, float addrspace(1)* %in.gep.0, i32 1

  %a = load volatile float, float addrspace(1)* %in.gep.0, align 4
  %b = load volatile float, float addrspace(1)* %in.gep.1, align 4
  %c = load volatile float, float addrspace(1)* %in.gep.2, align 4

  %mul0 = fmul float %a, %b
  %mul1 = fmul float %a, %c
  %madak0 = fadd float %mul0, 10.0
  %madak1 = fadd float %mul1, 10.0

  store volatile float %madak0, float addrspace(1)* %out.gep.0, align 4
  store volatile float %madak1, float addrspace(1)* %out.gep.1, align 4
  ret void
}

; GCN-LABEL: {{^}}madak_m_inline_imm_f32:
; GCN:       {{buffer|flat|global}}_load_{{dword|b32}} [[VA:v[0-9]+]]
; MAD:       v_madak_f32 {{v[0-9]+}}, 4.0, [[VA]], 0x41200000
; GFX10-MAD: v_madak_f32 {{v[0-9]+}}, 4.0, [[VA]], 0x41200000
; FMA:       v_fmaak_f32 {{v[0-9]+}}, 4.0, [[VA]], 0x41200000
; GFX11-MAD: v_mul_f32_e32 [[VMUL:v[0-9]+]], 4.0, [[VA]]
; GFX11-MAD: v_add_f32_e32 {{v[0-9]+}}, 0x41200000, [[VMUL]]
define amdgpu_kernel void @madak_m_inline_imm_f32(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in.a) #0 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %in.a.gep = getelementptr float, float addrspace(1)* %in.a, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid

  %a = load float, float addrspace(1)* %in.a.gep, align 4

  %mul = fmul float 4.0, %a
  %madak = fadd float %mul, 10.0
  store float %madak, float addrspace(1)* %out.gep, align 4
  ret void
}

; Make sure nothing weird happens with a value that is also allowed as
; an inline immediate.

; GCN-LABEL: {{^}}madak_inline_imm_f32:
; GFX6:      buffer_load_dword [[VA:v[0-9]+]]
; GFX6:      buffer_load_dword [[VB:v[0-9]+]]
; GFX8:      {{flat|global}}_load_dword [[VA:v[0-9]+]]
; GFX8:      {{flat|global}}_load_dword [[VB:v[0-9]+]]
; GFX9:      {{flat|global}}_load_dword [[VA:v[0-9]+]]
; GFX9:      {{flat|global}}_load_dword [[VB:v[0-9]+]]
; GFX10PLUS: {{flat|global}}_load_{{dword|b32}} [[VA:v[0-9]+]]
; GFX10PLUS: {{flat|global}}_load_{{dword|b32}} [[VB:v[0-9]+]]
; MAD:       v_mad_f32 {{v[0-9]+}}, [[VA]], [[VB]], 4.0
; GFX10-MAD: v_mad_f32 {{v[0-9]+}}, [[VA]], [[VB]], 4.0
; FMA:       v_fma_f32 {{v[0-9]+}}, [[VA]], [[VB]], 4.0
; GFX11-MAD: v_mul_f32_e32 [[VMUL:v[0-9]+]], [[VA]], [[VB]]
; GFX11-MAD: v_add_f32_e32 {{v[0-9]+}}, 4.0, [[VMUL]]
define amdgpu_kernel void @madak_inline_imm_f32(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in.a, float addrspace(1)* noalias %in.b) #0 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %in.a.gep = getelementptr float, float addrspace(1)* %in.a, i32 %tid
  %in.b.gep = getelementptr float, float addrspace(1)* %in.b, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid

  %a = load float, float addrspace(1)* %in.a.gep, align 4
  %b = load float, float addrspace(1)* %in.b.gep, align 4

  %mul = fmul float %a, %b
  %madak = fadd float %mul, 4.0
  store float %madak, float addrspace(1)* %out.gep, align 4
  ret void
}

; We can't use an SGPR when forming madak
; GCN-LABEL: {{^}}s_v_madak_f32:
; GCN-DAG:      s_load_{{dword|b32}} [[SB:s[0-9]+]]
; GFX6_8_9-DAG: v_mov_b32_e32 [[VK:v[0-9]+]], 0x41200000
; GCN-DAG:      {{buffer|flat|global}}_load_{{dword|b32}}{{(_addtid)?}} [[VA:v[0-9]+]]
; GCN-NOT:      v_madak_f32
; GFX6_8_9:     v_mac_f32_e32 [[VK]], [[SB]], [[VA]]
; GFX10-MAD:    v_mad_f32 v{{[0-9]+}}, [[VA]], [[SB]], 0x41200000
; FMA:          v_fma_f32 v{{[0-9]+}}, [[VA]], [[SB]], 0x41200000
; GFX11-MAD:    v_mul_f32_e32 [[VMUL:v[0-9]+]], [[SB]], [[VA]]
; GFX11-MAD:    v_add_f32_e32 {{v[0-9]+}}, 0x41200000, [[VMUL]]
define amdgpu_kernel void @s_v_madak_f32(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in.a, float %b) #0 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %in.a.gep = getelementptr float, float addrspace(1)* %in.a, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid

  %a = load float, float addrspace(1)* %in.a.gep, align 4

  %mul = fmul float %a, %b
  %madak = fadd float %mul, 10.0
  store float %madak, float addrspace(1)* %out.gep, align 4
  ret void
}

; GCN-LABEL: @v_s_madak_f32
; GCN-DAG:       s_load_{{dword|b32}} [[SB:s[0-9]+]]
; GFX6_8_9-DAG:  v_mov_b32_e32 [[VK:v[0-9]+]], 0x41200000
; GCN-DAG:       {{buffer|flat|global}}_load_{{dword|b32}}{{(_addtid)?}} [[VA:v[0-9]+]]
; GFX6_8_9-NOT:  v_madak_f32
; GFX6_8_9:      v_mac_f32_e32 [[VK]], [[SB]], [[VA]]
; GFX10-MAD:     v_madak_f32 v{{[0-9]+}}, [[SB]], [[VA]], 0x41200000
; FMA:           v_fmaak_f32 v{{[0-9]+}}, [[SB]], [[VA]], 0x41200000
; GFX11-MAD:     v_mul_f32_e32 [[VMUL:v[0-9]+]], [[SB]], [[VA]]
; GFX11-MAD:     v_add_f32_e32 {{v[0-9]+}}, 0x41200000, [[VMUL]]
define amdgpu_kernel void @v_s_madak_f32(float addrspace(1)* noalias %out, float %a, float addrspace(1)* noalias %in.b) #0 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %in.b.gep = getelementptr float, float addrspace(1)* %in.b, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid

  %b = load float, float addrspace(1)* %in.b.gep, align 4

  %mul = fmul float %a, %b
  %madak = fadd float %mul, 10.0
  store float %madak, float addrspace(1)* %out.gep, align 4
  ret void
}

; GCN-LABEL: {{^}}s_s_madak_f32:
; GCN-NOT:   v_madak_f32
; GFX8_9:    v_mac_f32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
; GFX10-MAD: v_mac_f32_e64 {{v[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; FMA:       v_fmac_f32_e64 {{v[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; GFX11-MAD: v_mul_f32_e64 [[VMUL:v[0-9]+]], {{s[0-9]+}}, {{s[0-9]+}}
; GFX11-MAD: v_dual_mov_b32 {{v[0-9]+}}, 0 :: v_dual_add_f32 {{v[0-9]+}}, 0x41200000, [[VMUL]]
define amdgpu_kernel void @s_s_madak_f32(float addrspace(1)* %out, float %a, float %b) #0 {
  %mul = fmul float %a, %b
  %madak = fadd float %mul, 10.0
  store float %madak, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}no_madak_src0_modifier_f32:
; GFX6:      buffer_load_dword [[VA:v[0-9]+]]
; GFX6:      buffer_load_dword [[VB:v[0-9]+]]
; GFX8PLUS:  {{flat|global}}_load_{{dword|b32}} [[VB:v[0-9]+]]
; GFX8PLUS:  {{flat|global}}_load_{{dword|b32}} [[VA:v[0-9]+]]
; GFX6_8_9:  v_mad_f32 {{v[0-9]+}}, |{{v[0-9]+}}|, {{v[0-9]+}}, {{[sv][0-9]+}}
; GFX10-MAD: v_mad_f32 {{v[0-9]+}}, |{{v[0-9]+}}|, {{v[0-9]+}}, 0x41200000
; FMA:       v_fma_f32 {{v[0-9]+}}, |{{v[0-9]+}}|, {{v[0-9]+}}, 0x41200000
; GFX11-MAD: v_mul_f32_e64 [[VMUL:v[0-9]+]], |{{v[0-9]+}}|, {{v[0-9]+}}
; GFX11-MAD: v_add_f32_e32 {{v[0-9]+}}, 0x41200000, [[VMUL]]
; GCN:       s_endpgm
define amdgpu_kernel void @no_madak_src0_modifier_f32(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in.a, float addrspace(1)* noalias %in.b) #0 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %in.a.gep = getelementptr float, float addrspace(1)* %in.a, i32 %tid
  %in.b.gep = getelementptr float, float addrspace(1)* %in.b, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid

  %a = load float, float addrspace(1)* %in.a.gep, align 4
  %b = load float, float addrspace(1)* %in.b.gep, align 4

  %a.fabs = call float @llvm.fabs.f32(float %a) nounwind readnone

  %mul = fmul float %a.fabs, %b
  %madak = fadd float %mul, 10.0
  store float %madak, float addrspace(1)* %out.gep, align 4
  ret void
}

; GCN-LABEL: {{^}}no_madak_src1_modifier_f32:
; GFX6:      buffer_load_dword [[VA:v[0-9]+]]
; GFX6:      buffer_load_dword [[VB:v[0-9]+]]
; GFX8PLUS:  {{flat|global}}_load_{{dword|b32}} [[VB:v[0-9]+]]
; GFX8PLUS:  {{flat|global}}_load_{{dword|b32}} [[VA:v[0-9]+]]
; GFX6_8_9:  v_mad_f32 {{v[0-9]+}}, {{v[0-9]+}}, |{{v[0-9]+}}|, {{[sv][0-9]+}}
; GFX10-MAD: v_mad_f32 {{v[0-9]+}}, {{v[0-9]+}}, |{{v[0-9]+}}|, 0x41200000
; FMA:       v_fma_f32 {{v[0-9]+}}, {{v[0-9]+}}, |{{v[0-9]+}}|, 0x41200000
; GFX11-MAD: v_mul_f32_e64 [[VMUL:v[0-9]+]], {{v[0-9]+}}, |{{v[0-9]+}}|
; GFX11-MAD: v_add_f32_e32 {{v[0-9]+}}, 0x41200000, [[VMUL]]
; GCN:       s_endpgm
define amdgpu_kernel void @no_madak_src1_modifier_f32(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in.a, float addrspace(1)* noalias %in.b) #0 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %in.a.gep = getelementptr float, float addrspace(1)* %in.a, i32 %tid
  %in.b.gep = getelementptr float, float addrspace(1)* %in.b, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid

  %a = load float, float addrspace(1)* %in.a.gep, align 4
  %b = load float, float addrspace(1)* %in.b.gep, align 4

  %b.fabs = call float @llvm.fabs.f32(float %b) nounwind readnone

  %mul = fmul float %a, %b.fabs
  %madak = fadd float %mul, 10.0
  store float %madak, float addrspace(1)* %out.gep, align 4
  ret void
}

; SIFoldOperands should not fold the SGPR copy into the instruction before GFX10
; because the implicit immediate already uses the constant bus.
; On GFX10+ we can use two scalar operands.
; GCN-LABEL: {{^}}madak_constant_bus_violation:
; GCN:       s_load_{{dword|b32}} [[SGPR0:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, {{0x12|0x48}}

; GCN:       {{buffer|flat|global}}_load_{{dword|b32}} [[VGPR:v[0-9]+]]
; MAD:       v_mov_b32_e32 [[MADAK:v[0-9]+]], 0x42280000
; MAD:       v_mac_f32_e64 [[MADAK]], [[SGPR0]], 0.5
; GFX10PLUS: v_mov_b32_e32 [[SGPR0_VCOPY:v[0-9]+]], [[SGPR0]]
; GFX10-MAD: v_madak_f32 [[MADAK:v[0-9]+]], 0.5, [[SGPR0_VCOPY]], 0x42280000
; FMA:       v_fmaak_f32 [[MADAK:v[0-9]+]], 0.5, [[SGPR0_VCOPY]], 0x42280000
; GFX11-MAD: v_mul_f32_e64 [[VMUL:v[0-9]+]], [[SGPR0]], 0.5
; GFX11-MAD: v_add_f32_e32 [[MADAK:v[0-9]+]], 0x42280000, [[VMUL]]
; GCN:       v_mul_f32_e32 [[MUL:v[0-9]+]], [[MADAK]], [[VGPR]]
; GFX6:      buffer_store_dword [[MUL]]
; GFX8PLUS:  {{flat|global}}_store_{{dword|b32}} v[{{[0-9:]+}}], [[MUL]]
define amdgpu_kernel void @madak_constant_bus_violation(i32 %arg1, [8 x i32], float %sgpr0, float %sgpr1) #0 {
bb:
  %tmp = icmp eq i32 %arg1, 0
  br i1 %tmp, label %bb3, label %bb4

bb3:
  store volatile float 0.0, float addrspace(1)* undef
  br label %bb4

bb4:
  %vgpr = load volatile float, float addrspace(1)* undef
  %tmp0 = fmul float %sgpr0, 0.5
  %tmp1 = fadd float %tmp0, 42.0
  %tmp2 = fmul float %tmp1, %vgpr
  store volatile float %tmp2, float addrspace(1)* undef, align 4
  ret void
}

attributes #0 = { nounwind "denormal-fp-math-f32"="preserve-sign,preserve-sign" }
