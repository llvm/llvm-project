; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs -enable-no-signed-zeros-fp-math < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs -enable-no-signed-zeros-fp-math < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; GCN-LABEL: {{^}}add_select_fabs_fabs_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]
; GCN: buffer_load_dword [[Z:v[0-9]+]]

; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], [[Y]], [[X]], vcc
; GCN: v_add_f32_e64 v{{[0-9]+}}, |[[SELECT]]|, [[Z]]
define void @add_select_fabs_fabs_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %z = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fabs.x = call float @llvm.fabs.f32(float %x)
  %fabs.y = call float @llvm.fabs.f32(float %y)
  %select = select i1 %cmp, float %fabs.x, float %fabs.y
  %add = fadd float %select, %z
  store float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}add_select_multi_use_lhs_fabs_fabs_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]
; GCN: buffer_load_dword [[Z:v[0-9]+]]
; GCN: buffer_load_dword [[W:v[0-9]+]]

; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], [[Y]], [[X]], vcc
; GCN-DAG: v_add_f32_e64 v{{[0-9]+}}, |[[SELECT]]|, [[Z]]
; GCN-DAG: v_add_f32_e64 v{{[0-9]+}}, |[[X]]|, [[W]]
define void @add_select_multi_use_lhs_fabs_fabs_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %z = load volatile float, float addrspace(1)* undef
  %w = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fabs.x = call float @llvm.fabs.f32(float %x)
  %fabs.y = call float @llvm.fabs.f32(float %y)
  %select = select i1 %cmp, float %fabs.x, float %fabs.y
  %add0 = fadd float %select, %z
  %add1 = fadd float %fabs.x, %w
  store volatile float %add0, float addrspace(1)* undef
  store volatile float %add1, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}add_select_multi_store_use_lhs_fabs_fabs_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]
; GCN: buffer_load_dword [[Z:v[0-9]+]]

; GCN-DAG: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], [[Y]], [[X]], vcc
; GCN-DAG: v_add_f32_e64 [[ADD:v[0-9]+]], |[[SELECT]]|, [[Z]]
; GCN-DAG: v_and_b32_e32 [[X_ABS:v[0-9]+]], 0x7fffffff, [[X]]

; GCN: buffer_store_dword [[ADD]]
; GCN: buffer_store_dword [[X_ABS]]
define void @add_select_multi_store_use_lhs_fabs_fabs_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %z = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fabs.x = call float @llvm.fabs.f32(float %x)
  %fabs.y = call float @llvm.fabs.f32(float %y)
  %select = select i1 %cmp, float %fabs.x, float %fabs.y
  %add0 = fadd float %select, %z
  store volatile float %add0, float addrspace(1)* undef
  store volatile float %fabs.x, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}add_select_multi_use_rhs_fabs_fabs_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]
; GCN: buffer_load_dword [[Z:v[0-9]+]]
; GCN: buffer_load_dword [[W:v[0-9]+]]

; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], [[Y]], [[X]], vcc
; GCN-DAG: v_add_f32_e64 v{{[0-9]+}}, |[[SELECT]]|, [[Z]]
; GCN-DAG: v_add_f32_e64 v{{[0-9]+}}, |[[Y]]|, [[W]]
define void @add_select_multi_use_rhs_fabs_fabs_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %z = load volatile float, float addrspace(1)* undef
  %w = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fabs.x = call float @llvm.fabs.f32(float %x)
  %fabs.y = call float @llvm.fabs.f32(float %y)
  %select = select i1 %cmp, float %fabs.x, float %fabs.y
  %add0 = fadd float %select, %z
  %add1 = fadd float %fabs.y, %w
  store volatile float %add0, float addrspace(1)* undef
  store volatile float %add1, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}add_select_fabs_var_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]
; GCN: buffer_load_dword [[Z:v[0-9]+]]

; GCN: v_and_b32_e32 [[X_ABS:v[0-9]+]], 0x7fffffff, [[X]]
; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], [[Y]], [[X_ABS]], vcc
; GCN: v_add_f32_e32 v{{[0-9]+}}, [[Z]], [[SELECT]]
define void @add_select_fabs_var_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %z = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fabs.x = call float @llvm.fabs.f32(float %x)
  %select = select i1 %cmp, float %fabs.x, float %y
  %add = fadd float %select, %z
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}add_select_fabs_negk_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]

; GCN: v_and_b32_e32 [[FABS_X:v[0-9]+]], 0x7fffffff, [[X]]
; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], -1.0, [[FABS_X]], vcc
; GCN: v_add_f32_e32 v{{[0-9]+}}, [[Y]], [[SELECT]]
define void @add_select_fabs_negk_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fabs = call float @llvm.fabs.f32(float %x)
  %select = select i1 %cmp, float %fabs, float -1.0
  %add = fadd float %select, %y
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; FIXME: fabs should fold away
; GCN-LABEL: {{^}}add_select_fabs_negk_negk_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]

; GCN: v_cndmask_b32_e64 [[SELECT:v[0-9]+]], -1.0, -2.0, s
; GCN: v_add_f32_e64 v{{[0-9]+}}, |[[SELECT]]|, [[X]]
define void @add_select_fabs_negk_negk_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, float -2.0, float -1.0
  %fabs = call float @llvm.fabs.f32(float %select)
  %add = fadd float %fabs, %x
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}add_select_posk_posk_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]

; GCN: v_cndmask_b32_e64 [[SELECT:v[0-9]+]], 1.0, 2.0, s
; GCN: v_add_f32_e32 v{{[0-9]+}}, [[X]], [[SELECT]]
define void @add_select_posk_posk_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, float 2.0, float 1.0
  %add = fadd float %select, %x
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}add_select_negk_fabs_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]

; GCN-DAG: v_and_b32_e32 [[FABS_X:v[0-9]+]], 0x7fffffff, [[X]]
; GCN-DAG: v_cmp_ne_u32_e64 vcc, s{{[0-9]+}}, 0
; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], -1.0, [[FABS_X]], vcc
; GCN: v_add_f32_e32 v{{[0-9]+}}, [[Y]], [[SELECT]]
define void @add_select_negk_fabs_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fabs = call float @llvm.fabs.f32(float %x)
  %select = select i1 %cmp, float -1.0, float %fabs
  %add = fadd float %select, %y
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}add_select_negliteralk_fabs_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]
; GCN-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0xc4800000

; GCN-DAG: v_and_b32_e32 [[FABS_X:v[0-9]+]], 0x7fffffff, [[X]]
; GCN-DAG: v_cmp_ne_u32_e64 vcc, s{{[0-9]+}}, 0
; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], [[K]], [[FABS_X]], vcc
; GCN: v_add_f32_e32 v{{[0-9]+}}, [[Y]], [[SELECT]]
define void @add_select_negliteralk_fabs_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fabs = call float @llvm.fabs.f32(float %x)
  %select = select i1 %cmp, float -1024.0, float %fabs
  %add = fadd float %select, %y
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}add_select_fabs_posk_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]

; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], 1.0, [[X]], vcc
; GCN: v_add_f32_e64 v{{[0-9]+}}, |[[SELECT]]|, [[Y]]
define void @add_select_fabs_posk_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef

  %cmp = icmp eq i32 %c, 0
  %fabs = call float @llvm.fabs.f32(float %x)
  %select = select i1 %cmp, float %fabs, float 1.0
  %add = fadd float %select, %y
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}add_select_posk_fabs_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]

; GCN: v_cmp_ne_u32_e64 vcc, s{{[0-9]+}}, 0
; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], 1.0, [[X]], vcc
; GCN: v_add_f32_e64 v{{[0-9]+}}, |[[SELECT]]|, [[Y]]
define void @add_select_posk_fabs_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fabs = call float @llvm.fabs.f32(float %x)
  %select = select i1 %cmp, float 1.0, float %fabs
  %add = fadd float %select, %y
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}add_select_fneg_fneg_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]
; GCN: buffer_load_dword [[Z:v[0-9]+]]

; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], [[Y]], [[X]], vcc
; GCN: v_subrev_f32_e32 v{{[0-9]+}}, [[SELECT]], [[Z]]
define void @add_select_fneg_fneg_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %z = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fneg.x = fsub float -0.0, %x
  %fneg.y = fsub float -0.0, %y
  %select = select i1 %cmp, float %fneg.x, float %fneg.y
  %add = fadd float %select, %z
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}add_select_multi_use_lhs_fneg_fneg_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]
; GCN: buffer_load_dword [[Z:v[0-9]+]]
; GCN: buffer_load_dword [[W:v[0-9]+]]

; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], [[Y]], [[X]], vcc
; GCN-DAG: v_subrev_f32_e32 v{{[0-9]+}}, [[SELECT]], [[Z]]
; GCN-DAG: v_subrev_f32_e32 v{{[0-9]+}}, [[X]], [[W]]
define void @add_select_multi_use_lhs_fneg_fneg_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %z = load volatile float, float addrspace(1)* undef
  %w = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fneg.x = fsub float -0.0, %x
  %fneg.y = fsub float -0.0, %y
  %select = select i1 %cmp, float %fneg.x, float %fneg.y
  %add0 = fadd float %select, %z
  %add1 = fadd float %fneg.x, %w
  store volatile float %add0, float addrspace(1)* undef
  store volatile float %add1, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}add_select_multi_store_use_lhs_fneg_fneg_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]
; GCN: buffer_load_dword [[Z:v[0-9]+]]

; GCN-DAG: v_xor_b32_e32 [[NEG_X:v[0-9]+]], 0x80000000, [[X]]
; GCN-DAG: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], [[Y]], [[X]], vcc
; GCN-DAG: v_subrev_f32_e32 [[ADD:v[0-9]+]], [[SELECT]], [[Z]]

; GCN: buffer_store_dword [[ADD]]
; GCN: buffer_store_dword [[NEG_X]]
define void @add_select_multi_store_use_lhs_fneg_fneg_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %z = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fneg.x = fsub float -0.0, %x
  %fneg.y = fsub float -0.0, %y
  %select = select i1 %cmp, float %fneg.x, float %fneg.y
  %add0 = fadd float %select, %z
  store volatile float %add0, float addrspace(1)* undef
  store volatile float %fneg.x, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}add_select_multi_use_rhs_fneg_fneg_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]
; GCN: buffer_load_dword [[Z:v[0-9]+]]
; GCN: buffer_load_dword [[W:v[0-9]+]]

; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], [[Y]], [[X]], vcc
; GCN-DAG: v_subrev_f32_e32 v{{[0-9]+}}, [[SELECT]], [[Z]]
; GCN-DAG: v_subrev_f32_e32 v{{[0-9]+}}, [[Y]], [[W]]
define void @add_select_multi_use_rhs_fneg_fneg_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %z = load volatile float, float addrspace(1)* undef
  %w = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fneg.x = fsub float -0.0, %x
  %fneg.y = fsub float -0.0, %y
  %select = select i1 %cmp, float %fneg.x, float %fneg.y
  %add0 = fadd float %select, %z
  %add1 = fadd float %fneg.y, %w
  store volatile float %add0, float addrspace(1)* undef
  store volatile float %add1, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}add_select_fneg_var_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]
; GCN: buffer_load_dword [[Z:v[0-9]+]]

; GCN: v_xor_b32_e32 [[X_NEG:v[0-9]+]], 0x80000000, [[X]]
; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], [[Y]], [[X_NEG]], vcc
; GCN: v_add_f32_e32 v{{[0-9]+}}, [[Z]], [[SELECT]]
define void @add_select_fneg_var_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %z = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fneg.x = fsub float -0.0, %x
  %select = select i1 %cmp, float %fneg.x, float %y
  %add = fadd float %select, %z
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}add_select_fneg_negk_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]

; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], 1.0, [[X]], vcc
; GCN: v_subrev_f32_e32 v{{[0-9]+}}, [[SELECT]], [[Y]]
define void @add_select_fneg_negk_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fneg.x = fsub float -0.0, %x
  %select = select i1 %cmp, float %fneg.x, float -1.0
  %add = fadd float %select, %y
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}add_select_fneg_inv2pi_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 0xbe22f983

; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], [[K]], [[X]], vcc
; GCN: v_subrev_f32_e32 v{{[0-9]+}}, [[SELECT]], [[Y]]
define void @add_select_fneg_inv2pi_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fneg.x = fsub float -0.0, %x
  %select = select i1 %cmp, float %fneg.x, float 0x3FC45F3060000000
  %add = fadd float %select, %y
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}add_select_fneg_neginv2pi_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]
; SI: v_mov_b32_e32 [[K:v[0-9]+]], 0x3e22f983

; SI: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], [[K]], [[X]], vcc
; VI: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], 0.15915494, [[X]], vcc

; GCN: v_subrev_f32_e32 v{{[0-9]+}}, [[SELECT]], [[Y]]
define void @add_select_fneg_neginv2pi_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fneg.x = fsub float -0.0, %x
  %select = select i1 %cmp, float %fneg.x, float 0xBFC45F3060000000
  %add = fadd float %select, %y
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}add_select_negk_negk_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]

; GCN: v_cmp_eq_u32_e64
; GCN: v_cndmask_b32_e64 [[SELECT:v[0-9]+]], -1.0, -2.0, s
; GCN: v_add_f32_e32 v{{[0-9]+}}, [[X]], [[SELECT]]
define void @add_select_negk_negk_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, float -2.0, float -1.0
  %add = fadd float %select, %x
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}add_select_negliteralk_negliteralk_f32:
; GCN-DAG: v_mov_b32_e32 [[K0:v[0-9]+]], 0xc5000000
; GCN-DAG: v_mov_b32_e32 [[K1:v[0-9]+]], 0xc5800000
; GCN-DAG: buffer_load_dword [[X:v[0-9]+]]

; GCN: v_cmp_eq_u32_e64
; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], [[K1]], [[K0]], vcc
; GCN: v_add_f32_e32 v{{[0-9]+}}, [[X]], [[SELECT]]
define void @add_select_negliteralk_negliteralk_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, float -2048.0, float -4096.0
  %add = fadd float %select, %x
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}add_select_fneg_negk_negk_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]

; GCN: v_cndmask_b32_e64 [[SELECT:v[0-9]+]], -1.0, -2.0, s
; GCN: v_subrev_f32_e32 v{{[0-9]+}}, [[SELECT]], [[X]]
define void @add_select_fneg_negk_negk_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, float -2.0, float -1.0
  %fneg.x = fsub float -0.0, %select
  %add = fadd float %fneg.x, %x
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}add_select_negk_fneg_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]

; GCN: v_cmp_ne_u32_e64 vcc, s{{[0-9]+}}, 0
; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], 1.0, [[X]], vcc
; GCN: v_subrev_f32_e32 v{{[0-9]+}}, [[SELECT]], [[Y]]
define void @add_select_negk_fneg_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fneg.x = fsub float -0.0, %x
  %select = select i1 %cmp, float -1.0, float %fneg.x
  %add = fadd float %select, %y
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}add_select_fneg_posk_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]

; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], -1.0, [[X]], vcc
; GCN: v_subrev_f32_e32 v{{[0-9]+}}, [[SELECT]], [[Y]]
define void @add_select_fneg_posk_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fneg.x = fsub float -0.0, %x
  %select = select i1 %cmp, float %fneg.x, float 1.0
  %add = fadd float %select, %y
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}add_select_posk_fneg_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]

; GCN: v_cmp_ne_u32_e64 vcc, s{{[0-9]+}}, 0
; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], -1.0, [[X]], vcc
; GCN: v_subrev_f32_e32 v{{[0-9]+}}, [[SELECT]], [[Y]]
define void @add_select_posk_fneg_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fneg.x = fsub float -0.0, %x
  %select = select i1 %cmp, float 1.0, float %fneg.x
  %add = fadd float %select, %y
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}add_select_negfabs_fabs_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]
; GCN: buffer_load_dword [[Z:v[0-9]+]]

; GCN-DAG: v_or_b32_e32 [[X_NEG_ABS:v[0-9]+]], 0x80000000, [[X]]
; GCN-DAG: v_and_b32_e32 [[Y_ABS:v[0-9]+]], 0x7fffffff, [[Y]]
; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], [[Y_ABS]], [[X_NEG_ABS]], vcc
; GCN: v_add_f32_e32 v{{[0-9]+}}, [[Z]], [[SELECT]]
define void @add_select_negfabs_fabs_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %z = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fabs.x = call float @llvm.fabs.f32(float %x)
  %fneg.fabs.x = fsub float -0.000000e+00, %fabs.x
  %fabs.y = call float @llvm.fabs.f32(float %y)
  %select = select i1 %cmp, float %fneg.fabs.x, float %fabs.y
  %add = fadd float %select, %z
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}add_select_fabs_negfabs_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]
; GCN: buffer_load_dword [[Z:v[0-9]+]]

; GCN-DAG: v_or_b32_e32 [[Y_NEG_ABS:v[0-9]+]], 0x80000000, [[Y]]
; GCN-DAG: v_and_b32_e32 [[X_ABS:v[0-9]+]], 0x7fffffff, [[X]]
; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], [[Y_NEG_ABS]], [[X_ABS]], vcc
; GCN: v_add_f32_e32 v{{[0-9]+}}, [[Z]], [[SELECT]]
define void @add_select_fabs_negfabs_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %z = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fabs.x = call float @llvm.fabs.f32(float %x)
  %fabs.y = call float @llvm.fabs.f32(float %y)
  %fneg.fabs.y = fsub float -0.000000e+00, %fabs.y
  %select = select i1 %cmp, float %fabs.x, float %fneg.fabs.y
  %add = fadd float %select, %z
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}add_select_neg_fabs_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]
; GCN: buffer_load_dword [[Z:v[0-9]+]]

; GCN-DAG: v_xor_b32_e32 [[X_NEG:v[0-9]+]], 0x80000000, [[X]]
; GCN-DAG: v_and_b32_e32 [[Y_ABS:v[0-9]+]], 0x7fffffff, [[Y]]
; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], [[Y_ABS]], [[X_NEG]], vcc
; GCN: v_add_f32_e32 v{{[0-9]+}}, [[Z]], [[SELECT]]
define void @add_select_neg_fabs_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %z = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fneg.x = fsub float -0.000000e+00, %x
  %fabs.y = call float @llvm.fabs.f32(float %y)
  %select = select i1 %cmp, float %fneg.x, float %fabs.y
  %add = fadd float %select, %z
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}add_select_fabs_neg_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]
; GCN: buffer_load_dword [[Z:v[0-9]+]]

; GCN-DAG: v_and_b32_e32 [[X_ABS:v[0-9]+]], 0x7fffffff, [[X]]
; GCN-DAG: v_xor_b32_e32 [[Y_NEG:v[0-9]+]], 0x80000000, [[Y]]
; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], [[Y_NEG]], [[X_ABS]], vcc
; GCN: v_add_f32_e32 v{{[0-9]+}}, [[Z]], [[SELECT]]
define void @add_select_fabs_neg_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %z = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fabs.x = call float @llvm.fabs.f32(float %x)
  %fneg.y = fsub float -0.000000e+00, %y
  %select = select i1 %cmp, float %fabs.x, float %fneg.y
  %add = fadd float %select, %z
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}add_select_neg_negfabs_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]
; GCN: buffer_load_dword [[Z:v[0-9]+]]

; GCN-DAG: v_and_b32_e32 [[Y_ABS:v[0-9]+]], 0x7fffffff, [[Y]]
; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], [[Y_ABS]], [[X]], vcc
; GCN: v_subrev_f32_e32 v{{[0-9]+}}, [[SELECT]], [[Z]]
define void @add_select_neg_negfabs_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %z = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fneg.x = fsub float -0.000000e+00, %x
  %fabs.y = call float @llvm.fabs.f32(float %y)
  %fneg.fabs.y = fsub float -0.000000e+00, %fabs.y
  %select = select i1 %cmp, float %fneg.x, float %fneg.fabs.y
  %add = fadd float %select, %z
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}add_select_negfabs_neg_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]
; GCN: buffer_load_dword [[Z:v[0-9]+]]

; GCN-DAG: v_and_b32_e32 [[X_ABS:v[0-9]+]], 0x7fffffff, [[X]]
; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], [[X_ABS]], [[Y]], vcc
; GCN: v_subrev_f32_e32 v{{[0-9]+}}, [[SELECT]], [[Z]]
define void @add_select_negfabs_neg_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %z = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fabs.x = call float @llvm.fabs.f32(float %x)
  %fneg.fabs.x = fsub float -0.000000e+00, %fabs.x
  %fneg.y = fsub float -0.000000e+00, %y
  %select = select i1 %cmp, float %fneg.y, float %fneg.fabs.x
  %add = fadd float %select, %z
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}mul_select_negfabs_posk_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]

; GCN-DAG: v_cmp_eq_u32_e64 vcc,
; GCN-DAG: v_and_b32_e32 [[X_ABS:v[0-9]+]], 0x7fffffff, [[X]]
; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], -4.0, [[X_ABS]], vcc
; GCN: v_mul_f32_e64 v{{[0-9]+}}, -[[SELECT]], [[Y]]
define void @mul_select_negfabs_posk_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fabs.x = call float @llvm.fabs.f32(float %x)
  %fneg.fabs.x = fsub float -0.000000e+00, %fabs.x
  %select = select i1 %cmp, float %fneg.fabs.x, float 4.0
  %add = fmul float %select, %y
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}mul_select_posk_negfabs_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]

; GCN-DAG: v_cmp_ne_u32_e64 vcc, s{{[0-9]+}}, 0
; GCN-DAG: v_and_b32_e32 [[X_ABS:v[0-9]+]], 0x7fffffff, [[X]]

; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], -4.0, [[X_ABS]], vcc
; GCN: v_mul_f32_e64 v{{[0-9]+}}, -[[SELECT]], [[Y]]
define void @mul_select_posk_negfabs_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fabs.x = call float @llvm.fabs.f32(float %x)
  %fneg.fabs.x = fsub float -0.000000e+00, %fabs.x
  %select = select i1 %cmp, float 4.0, float %fneg.fabs.x
  %add = fmul float %select, %y
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}mul_select_negfabs_negk_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]

; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], 4.0, [[X]], vcc
; GCN: v_mul_f32_e64 v{{[0-9]+}}, -|[[SELECT]]|, [[Y]]
define void @mul_select_negfabs_negk_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fabs.x = call float @llvm.fabs.f32(float %x)
  %fneg.fabs.x = fsub float -0.000000e+00, %fabs.x
  %select = select i1 %cmp, float %fneg.fabs.x, float -4.0
  %add = fmul float %select, %y
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}mul_select_negk_negfabs_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]

; GCN: v_cmp_ne_u32_e64 vcc
; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], 4.0, [[X]], vcc
; GCN: v_mul_f32_e64 v{{[0-9]+}}, -|[[SELECT]]|, [[Y]]
define void @mul_select_negk_negfabs_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fabs.x = call float @llvm.fabs.f32(float %x)
  %fneg.fabs.x = fsub float -0.000000e+00, %fabs.x
  %select = select i1 %cmp, float -4.0, float %fneg.fabs.x
  %add = fmul float %select, %y
  store volatile float %add, float addrspace(1)* undef
  ret void
}

; --------------------------------------------------------------------------------
; Don't fold if fneg can fold into the source
; --------------------------------------------------------------------------------

; GCN-LABEL: {{^}}select_fneg_posk_src_add_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Y:v[0-9]+]]

; GCN: v_sub_f32_e32 [[ADD:v[0-9]+]], -4.0, [[X]]
; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], 2.0, [[ADD]], vcc
; GCN-NEXT: buffer_store_dword [[SELECT]]
define void @select_fneg_posk_src_add_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %add = fadd float %x, 4.0
  %fneg = fsub float -0.0, %add
  %select = select i1 %cmp, float %fneg, float 2.0
  store volatile float %select, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}select_fneg_posk_src_sub_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]

; GCN: v_sub_f32_e32 [[ADD:v[0-9]+]], 4.0, [[X]]
; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], 2.0, [[ADD]], vcc
; GCN-NEXT: buffer_store_dword [[SELECT]]
define void @select_fneg_posk_src_sub_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %add = fsub float %x, 4.0
  %fneg = fsub float -0.0, %add
  %select = select i1 %cmp, float %fneg, float 2.0
  store volatile float %select, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}select_fneg_posk_src_mul_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]

; GCN: v_mul_f32_e32 [[MUL:v[0-9]+]], -4.0, [[X]]
; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], 2.0, [[MUL]], vcc
; GCN-NEXT: buffer_store_dword [[SELECT]]
define void @select_fneg_posk_src_mul_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %mul = fmul float %x, 4.0
  %fneg = fsub float -0.0, %mul
  %select = select i1 %cmp, float %fneg, float 2.0
  store volatile float %select, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}select_fneg_posk_src_fma_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Z:v[0-9]+]]

; GCN: v_fma_f32 [[FMA:v[0-9]+]], [[X]], -4.0, -[[Z]]
; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], 2.0, [[FMA]], vcc
; GCN-NEXT: buffer_store_dword [[SELECT]]
define void @select_fneg_posk_src_fma_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %z = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fma = call float @llvm.fma.f32(float %x, float 4.0, float %z)
  %fneg = fsub float -0.0, %fma
  %select = select i1 %cmp, float %fneg, float 2.0
  store volatile float %select, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}select_fneg_posk_src_fmad_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]
; GCN: buffer_load_dword [[Z:v[0-9]+]]

; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], 2.0, [[X]], vcc
; GCN-NEXT: buffer_store_dword [[SELECT]]
define void @select_fneg_posk_src_fmad_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %z = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %fmad = call float @llvm.fmuladd.f32(float %x, float 4.0, float %z)
  %fneg = fsub float -0.0, %fmad
  %select = select i1 %cmp, float %fneg, float 2.0
  store volatile float %select, float addrspace(1)* undef
  ret void
}

; FIXME: This one should fold to rcp
; GCN-LABEL: {{^}}select_fneg_posk_src_rcp_f32:
; GCN: buffer_load_dword [[X:v[0-9]+]]

; GCN: v_rcp_f32_e32 [[RCP:v[0-9]+]], [[X]]
; GCN: v_cndmask_b32_e32 [[SELECT:v[0-9]+]], -2.0, [[RCP]], vcc
; GCN: v_xor_b32_e32 [[NEG_SELECT:v[0-9]+]], 0x80000000, [[SELECT]]
; GCN-NEXT: buffer_store_dword [[NEG_SELECT]]
define void @select_fneg_posk_src_rcp_f32(i32 %c) #0 {
  %x = load volatile float, float addrspace(1)* undef
  %y = load volatile float, float addrspace(1)* undef
  %cmp = icmp eq i32 %c, 0
  %rcp = call float @llvm.amdgcn.rcp.f32(float %x)
  %fneg = fsub float -0.0, %rcp
  %select = select i1 %cmp, float %fneg, float 2.0
  store volatile float %select, float addrspace(1)* undef
  ret void
}

declare float @llvm.fabs.f32(float) #1
declare float @llvm.fma.f32(float, float, float) #1
declare float @llvm.fmuladd.f32(float, float, float) #1
declare float @llvm.amdgcn.rcp.f32(float) #1
declare float @llvm.amdgcn.rcp.legacy(float) #1
declare float @llvm.amdgcn.fmul.legacy(float, float) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
