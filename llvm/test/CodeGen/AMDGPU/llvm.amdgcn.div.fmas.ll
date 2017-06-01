; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -strict-whitespace -check-prefix=GCN -check-prefix=SI %s
; XUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -strict-whitespace -check-prefix=GCN -check-prefix=VI %s

; FIXME: Enable for VI.

declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
declare float @llvm.amdgcn.div.fmas.f32(float, float, float, i1) nounwind readnone
declare double @llvm.amdgcn.div.fmas.f64(double, double, double, i1) nounwind readnone

; GCN-LABEL: {{^}}test_div_fmas_f32:
; SI-DAG: s_load_dword [[SA:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: s_load_dword [[SC:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xd
; SI-DAG: s_load_dword [[SB:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xc
; VI-DAG: s_load_dword [[SA:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x2c
; VI-DAG: s_load_dword [[SC:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x34
; VI-DAG: s_load_dword [[SB:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x30
; GCN-DAG: v_mov_b32_e32 [[VC:v[0-9]+]], [[SC]]
; GCN-DAG: v_mov_b32_e32 [[VB:v[0-9]+]], [[SB]]
; GCN-DAG: v_mov_b32_e32 [[VA:v[0-9]+]], [[SA]]
; GCN: v_div_fmas_f32 [[RESULT:v[0-9]+]], [[VB]], [[VA]], [[VC]]
; GCN: buffer_store_dword [[RESULT]],
; GCN: s_endpgm
define void @test_div_fmas_f32(float addrspace(1)* %out, float %a, float %b, float %c, i1 %d) nounwind {
  %result = call float @llvm.amdgcn.div.fmas.f32(float %a, float %b, float %c, i1 %d) nounwind readnone
  store float %result, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_div_fmas_f32_inline_imm_0:
; SI-DAG: s_load_dword [[SC:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xd
; SI-DAG: s_load_dword [[SB:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xc
; SI-DAG: v_mov_b32_e32 [[VC:v[0-9]+]], [[SC]]
; SI-DAG: v_mov_b32_e32 [[VB:v[0-9]+]], [[SB]]
; SI: v_div_fmas_f32 [[RESULT:v[0-9]+]], 1.0, [[VB]], [[VC]]
; SI: buffer_store_dword [[RESULT]],
; SI: s_endpgm
define void @test_div_fmas_f32_inline_imm_0(float addrspace(1)* %out, float %a, float %b, float %c, i1 %d) nounwind {
  %result = call float @llvm.amdgcn.div.fmas.f32(float 1.0, float %b, float %c, i1 %d) nounwind readnone
  store float %result, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_div_fmas_f32_inline_imm_1:
; SI-DAG: s_load_dword [[SA:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: s_load_dword [[SC:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xd
; SI-DAG: v_mov_b32_e32 [[VC:v[0-9]+]], [[SC]]
; SI-DAG: v_mov_b32_e32 [[VA:v[0-9]+]], [[SA]]
; SI: v_div_fmas_f32 [[RESULT:v[0-9]+]], [[VA]], 1.0, [[VC]]
; SI: buffer_store_dword [[RESULT]],
; SI: s_endpgm
define void @test_div_fmas_f32_inline_imm_1(float addrspace(1)* %out, float %a, float %b, float %c, i1 %d) nounwind {
  %result = call float @llvm.amdgcn.div.fmas.f32(float %a, float 1.0, float %c, i1 %d) nounwind readnone
  store float %result, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_div_fmas_f32_inline_imm_2:
; SI-DAG: s_load_dword [[SA:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: s_load_dword [[SB:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xc
; SI-DAG: v_mov_b32_e32 [[VA:v[0-9]+]], [[SA]]
; SI-DAG: v_mov_b32_e32 [[VB:v[0-9]+]], [[SB]]
; SI: v_div_fmas_f32 [[RESULT:v[0-9]+]], [[VA]], [[VB]], 1.0
; SI: buffer_store_dword [[RESULT]],
; SI: s_endpgm
define void @test_div_fmas_f32_inline_imm_2(float addrspace(1)* %out, float %a, float %b, float %c, i1 %d) nounwind {
  %result = call float @llvm.amdgcn.div.fmas.f32(float %a, float %b, float 1.0, i1 %d) nounwind readnone
  store float %result, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_div_fmas_f64:
; GCN: v_div_fmas_f64
define void @test_div_fmas_f64(double addrspace(1)* %out, double %a, double %b, double %c, i1 %d) nounwind {
  %result = call double @llvm.amdgcn.div.fmas.f64(double %a, double %b, double %c, i1 %d) nounwind readnone
  store double %result, double addrspace(1)* %out, align 8
  ret void
}

; GCN-LABEL: {{^}}test_div_fmas_f32_cond_to_vcc:
; SI: v_cmp_eq_u32_e64 vcc, s{{[0-9]+}}, 0{{$}}
; SI: v_div_fmas_f32 {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
define void @test_div_fmas_f32_cond_to_vcc(float addrspace(1)* %out, float %a, float %b, float %c, i32 %i) nounwind {
  %cmp = icmp eq i32 %i, 0
  %result = call float @llvm.amdgcn.div.fmas.f32(float %a, float %b, float %c, i1 %cmp) nounwind readnone
  store float %result, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_div_fmas_f32_imm_false_cond_to_vcc:
; SI: s_mov_b64 vcc, 0
; SI: v_div_fmas_f32 {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
define void @test_div_fmas_f32_imm_false_cond_to_vcc(float addrspace(1)* %out, float %a, float %b, float %c) nounwind {
  %result = call float @llvm.amdgcn.div.fmas.f32(float %a, float %b, float %c, i1 false) nounwind readnone
  store float %result, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_div_fmas_f32_imm_true_cond_to_vcc:
; SI: s_mov_b64 vcc, -1
; SI: v_div_fmas_f32 {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
define void @test_div_fmas_f32_imm_true_cond_to_vcc(float addrspace(1)* %out, float %a, float %b, float %c) nounwind {
  %result = call float @llvm.amdgcn.div.fmas.f32(float %a, float %b, float %c, i1 true) nounwind readnone
  store float %result, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_div_fmas_f32_logical_cond_to_vcc:
; SI-DAG: buffer_load_dword [[A:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_load_dword [[B:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4{{$}}
; SI-DAG: buffer_load_dword [[C:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8{{$}}

; SI-DAG: v_cmp_eq_u32_e32 [[CMP0:vcc]], 0, v{{[0-9]+}}
; SI-DAG: v_cmp_ne_u32_e64 [[CMP1:s\[[0-9]+:[0-9]+\]]], s{{[0-9]+}}, 0{{$}}
; SI: s_and_b64 vcc, [[CMP0]], [[CMP1]]
; SI: v_div_fmas_f32 {{v[0-9]+}}, [[A]], [[B]], [[C]]
; SI: s_endpgm
define void @test_div_fmas_f32_logical_cond_to_vcc(float addrspace(1)* %out, float addrspace(1)* %in, i32 %d) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.a = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.b = getelementptr float, float addrspace(1)* %gep.a, i32 1
  %gep.c = getelementptr float, float addrspace(1)* %gep.a, i32 2
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 2

  %a = load volatile float, float addrspace(1)* %gep.a
  %b = load volatile float, float addrspace(1)* %gep.b
  %c = load volatile float, float addrspace(1)* %gep.c

  %cmp0 = icmp eq i32 %tid, 0
  %cmp1 = icmp ne i32 %d, 0
  %and = and i1 %cmp0, %cmp1

  %result = call float @llvm.amdgcn.div.fmas.f32(float %a, float %b, float %c, i1 %and) nounwind readnone
  store float %result, float addrspace(1)* %gep.out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_div_fmas_f32_i1_phi_vcc:
; SI: v_cmp_eq_u32_e32 vcc, 0, v{{[0-9]+}}
; SI: s_and_saveexec_b64 [[SAVE:s\[[0-9]+:[0-9]+\]]], vcc
; SI: s_xor_b64 [[SAVE]], exec, [[SAVE]]

; SI: buffer_load_dword [[LOAD:v[0-9]+]]
; SI: v_cmp_ne_u32_e32 vcc, 0, [[LOAD]]
; SI: v_cndmask_b32_e64 {{v[0-9]+}}, 0, -1, vcc


; SI: BB9_2:
; SI: s_or_b64 exec, exec, [[SAVE]]
; SI: v_cmp_ne_u32_e32 vcc, 0, v{{[0-9]+}}
; SI: v_div_fmas_f32 {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
; SI: buffer_store_dword
; SI: s_endpgm
define void @test_div_fmas_f32_i1_phi_vcc(float addrspace(1)* %out, float addrspace(1)* %in, i32 addrspace(1)* %dummy) nounwind {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 2
  %gep.a = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.b = getelementptr float, float addrspace(1)* %gep.a, i32 1
  %gep.c = getelementptr float, float addrspace(1)* %gep.a, i32 2

  %a = load float, float addrspace(1)* %gep.a
  %b = load float, float addrspace(1)* %gep.b
  %c = load float, float addrspace(1)* %gep.c

  %cmp0 = icmp eq i32 %tid, 0
  br i1 %cmp0, label %bb, label %exit

bb:
  %val = load i32, i32 addrspace(1)* %dummy
  %cmp1 = icmp ne i32 %val, 0
  br label %exit

exit:
  %cond = phi i1 [false, %entry], [%cmp1, %bb]
  %result = call float @llvm.amdgcn.div.fmas.f32(float %a, float %b, float %c, i1 %cond) nounwind readnone
  store float %result, float addrspace(1)* %gep.out, align 4
  ret void
}
