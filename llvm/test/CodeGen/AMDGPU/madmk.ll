; XFAIL: *
; RUN: llc -mtriple=amdgcn -mattr=+mad-mac-f32-insts -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; XUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

 ; FIXME: None of these trigger madmk emission anymore. It is still
 ; possible, but requires the correct registers to be used which is
 ; hard to trigger.

declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
declare float @llvm.fabs.f32(float) nounwind readnone

; GCN-LABEL: {{^}}madmk_f32:
; GCN-DAG: buffer_load_dword [[VA:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; GCN-DAG: buffer_load_dword [[VB:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; GCN: v_madmk_f32 {{v[0-9]+}}, [[VA]], 0x41200000, [[VB]]
define amdgpu_kernel void @madmk_f32(ptr addrspace(1) noalias %out, ptr addrspace(1) noalias %in) #0 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr float, ptr addrspace(1) %in, i32 %tid
  %gep.1 = getelementptr float, ptr addrspace(1) %gep.0, i32 1
  %out.gep = getelementptr float, ptr addrspace(1) %out, i32 %tid

  %a = load volatile float, ptr addrspace(1) %gep.0, align 4
  %b = load volatile float, ptr addrspace(1) %gep.1, align 4

  %mul = fmul float %a, 10.0
  %madmk = fadd float %mul, %b
  store float %madmk, ptr addrspace(1) %out.gep, align 4
  ret void
}

; GCN-LABEL: {{^}}madmk_2_use_f32:
; GCN-DAG: buffer_load_dword [[VA:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; GCN-DAG: buffer_load_dword [[VB:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; GCN-DAG: buffer_load_dword [[VC:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8
; GCN-DAG: v_mac_f32_e32 [[VB]], 0x41200000, [[VA]]
; GCN-DAG: v_mac_f32_e32 [[VC]], 0x41200000, [[VA]]
; GCN: s_endpgm
define amdgpu_kernel void @madmk_2_use_f32(ptr addrspace(1) noalias %out, ptr addrspace(1) noalias %in) #0 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone

  %in.gep.0 = getelementptr float, ptr addrspace(1) %in, i32 %tid
  %in.gep.1 = getelementptr float, ptr addrspace(1) %in.gep.0, i32 1
  %in.gep.2 = getelementptr float, ptr addrspace(1) %in.gep.0, i32 2

  %out.gep.0 = getelementptr float, ptr addrspace(1) %out, i32 %tid
  %out.gep.1 = getelementptr float, ptr addrspace(1) %in.gep.0, i32 1

  %a = load volatile float, ptr addrspace(1) %in.gep.0, align 4
  %b = load volatile float, ptr addrspace(1) %in.gep.1, align 4
  %c = load volatile float, ptr addrspace(1) %in.gep.2, align 4

  %mul0 = fmul float %a, 10.0
  %mul1 = fmul float %a, 10.0
  %madmk0 = fadd float %mul0, %b
  %madmk1 = fadd float %mul1, %c

  store float %madmk0, ptr addrspace(1) %out.gep.0, align 4
  store float %madmk1, ptr addrspace(1) %out.gep.1, align 4
  ret void
}

; We don't get any benefit if the constant is an inline immediate.
; GCN-LABEL: {{^}}madmk_inline_imm_f32:
; GCN-DAG: buffer_load_dword [[VA:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; GCN-DAG: buffer_load_dword [[VB:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; GCN: v_mac_f32_e32 [[VB]], 4.0, [[VA]]
define amdgpu_kernel void @madmk_inline_imm_f32(ptr addrspace(1) noalias %out, ptr addrspace(1) noalias %in) #0 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr float, ptr addrspace(1) %in, i32 %tid
  %gep.1 = getelementptr float, ptr addrspace(1) %gep.0, i32 1
  %out.gep = getelementptr float, ptr addrspace(1) %out, i32 %tid

  %a = load volatile float, ptr addrspace(1) %gep.0, align 4
  %b = load volatile float, ptr addrspace(1) %gep.1, align 4

  %mul = fmul float %a, 4.0
  %madmk = fadd float %mul, %b
  store float %madmk, ptr addrspace(1) %out.gep, align 4
  ret void
}

; GCN-LABEL: {{^}}s_s_madmk_f32:
; GCN-NOT: v_madmk_f32
; GCN: v_mac_f32_e32
; GCN: s_endpgm
define amdgpu_kernel void @s_s_madmk_f32(ptr addrspace(1) noalias %out, [8 x i32], float %a, [8 x i32], float %b) #0 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %out.gep = getelementptr float, ptr addrspace(1) %out, i32 %tid

  %mul = fmul float %a, 10.0
  %madmk = fadd float %mul, %b
  store float %madmk, ptr addrspace(1) %out.gep, align 4
  ret void
}

; GCN-LABEL: {{^}}v_s_madmk_f32:
; GCN-DAG: s_load_dword [[SREG:s[0-9]+]]
; GCN-DAG: buffer_load_dword [[VREG1:v[0-9]+]]
; GCN: v_mov_b32_e32 [[VREG2:v[0-9]+]], [[SREG]]
; GCN: v_madmk_f32 {{v[0-9]+}}, [[VREG1]], 0x41200000, [[VREG2]]
; GCN: s_endpgm
define amdgpu_kernel void @v_s_madmk_f32(ptr addrspace(1) noalias %out, ptr addrspace(1) noalias %in, float %b) #0 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr float, ptr addrspace(1) %in, i32 %tid
  %out.gep = getelementptr float, ptr addrspace(1) %out, i32 %tid
  %a = load float, ptr addrspace(1) %gep.0, align 4

  %mul = fmul float %a, 10.0
  %madmk = fadd float %mul, %b
  store float %madmk, ptr addrspace(1) %out.gep, align 4
  ret void
}

; GCN-LABEL: {{^}}scalar_vector_madmk_f32:
; GCN-NOT: v_madmk_f32
; GCN: v_mac_f32_e32
; GCN: s_endpgm
define amdgpu_kernel void @scalar_vector_madmk_f32(ptr addrspace(1) noalias %out, ptr addrspace(1) noalias %in, float %a) #0 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr float, ptr addrspace(1) %in, i32 %tid
  %out.gep = getelementptr float, ptr addrspace(1) %out, i32 %tid
  %b = load float, ptr addrspace(1) %gep.0, align 4

  %mul = fmul float %a, 10.0
  %madmk = fadd float %mul, %b
  store float %madmk, ptr addrspace(1) %out.gep, align 4
  ret void
}

; GCN-LABEL: {{^}}no_madmk_src0_modifier_f32:
; GCN-DAG: buffer_load_dword [[VA:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; GCN-DAG: buffer_load_dword [[VB:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; GCN-DAG: s_mov_b32 [[SK:s[0-9]+]], 0x41200000
; GCN: v_mad_f32 {{v[0-9]+}}, |[[VA]]|, [[SK]], [[VB]]
define amdgpu_kernel void @no_madmk_src0_modifier_f32(ptr addrspace(1) noalias %out, ptr addrspace(1) noalias %in) #0 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr float, ptr addrspace(1) %in, i32 %tid
  %gep.1 = getelementptr float, ptr addrspace(1) %gep.0, i32 1
  %out.gep = getelementptr float, ptr addrspace(1) %out, i32 %tid

  %a = load volatile float, ptr addrspace(1) %gep.0, align 4
  %b = load volatile float, ptr addrspace(1) %gep.1, align 4

  %a.fabs = call float @llvm.fabs.f32(float %a) nounwind readnone

  %mul = fmul float %a.fabs, 10.0
  %madmk = fadd float %mul, %b
  store float %madmk, ptr addrspace(1) %out.gep, align 4
  ret void
}

; GCN-LABEL: {{^}}no_madmk_src2_modifier_f32:
; GCN-DAG: buffer_load_dword [[VA:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; GCN-DAG: buffer_load_dword [[VB:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; GCN: v_mad_f32 {{v[0-9]+}}, {{v[0-9]+}}, {{[sv][0-9]+}}, |{{v[0-9]+}}|
define amdgpu_kernel void @no_madmk_src2_modifier_f32(ptr addrspace(1) noalias %out, ptr addrspace(1) noalias %in) #0 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr float, ptr addrspace(1) %in, i32 %tid
  %gep.1 = getelementptr float, ptr addrspace(1) %gep.0, i32 1
  %out.gep = getelementptr float, ptr addrspace(1) %out, i32 %tid

  %a = load volatile float, ptr addrspace(1) %gep.0, align 4
  %b = load volatile float, ptr addrspace(1) %gep.1, align 4

  %b.fabs = call float @llvm.fabs.f32(float %b) nounwind readnone

  %mul = fmul float %a, 10.0
  %madmk = fadd float %mul, %b.fabs
  store float %madmk, ptr addrspace(1) %out.gep, align 4
  ret void
}

; GCN-LABEL: {{^}}madmk_add_inline_imm_f32:
; GCN: buffer_load_dword [[A:v[0-9]+]]
; GCN: v_mov_b32_e32 [[B:v[0-9]+]], 2.0
; GCN: v_madmk_f32 {{v[0-9]+}}, [[A]], 0x41200000, [[B]]

define amdgpu_kernel void @madmk_add_inline_imm_f32(ptr addrspace(1) noalias %out, ptr addrspace(1) noalias %in) #0 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr float, ptr addrspace(1) %in, i32 %tid
  %out.gep = getelementptr float, ptr addrspace(1) %out, i32 %tid

  %a = load float, ptr addrspace(1) %gep.0, align 4

  %mul = fmul float %a, 10.0
  %madmk = fadd float %mul, 2.0
  store float %madmk, ptr addrspace(1) %out.gep, align 4
  ret void
}

; SI-LABEL: {{^}}kill_madmk_verifier_error:
; SI: s_or_b64
; SI: s_xor_b64
; SI: v_mac_f32_e32 {{v[0-9]+}}, 0x472aee8c, {{v[0-9]+}}
define amdgpu_kernel void @kill_madmk_verifier_error() #0 {
bb:
  br label %bb2

bb1:                                              ; preds = %bb2
  ret void

bb2:                                              ; preds = %bb6, %bb
  %tmp = phi float [ undef, %bb ], [ %tmp8, %bb6 ]
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #1
  %f_tid = bitcast i32 %tid to float
  %tmp3 = fsub float %f_tid, %tmp
  %tmp5 = fcmp oeq float %tmp3, 1.000000e+04
  br i1 %tmp5, label %bb1, label %bb6

bb6:                                              ; preds = %bb2
  %tmp7 = fmul float %tmp, 0x40E55DD180000000
  %tmp8 = fadd float %tmp7, %tmp
  br label %bb2
}

declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #1

attributes #0 = { nounwind "denormal-fp-math-f32"="preserve-sign,preserve-sign" }
attributes #1 = { nounwind readnone }
