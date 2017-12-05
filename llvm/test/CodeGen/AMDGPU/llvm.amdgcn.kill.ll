; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=SI %s

; SI-LABEL: {{^}}gs_const:
; SI-NOT: v_cmpx
; SI: s_mov_b64 exec, 0
define amdgpu_gs void @gs_const() {
  %tmp = icmp ule i32 0, 3
  %tmp1 = select i1 %tmp, float 1.000000e+00, float -1.000000e+00
  %c1 = fcmp oge float %tmp1, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  %tmp2 = icmp ule i32 3, 0
  %tmp3 = select i1 %tmp2, float 1.000000e+00, float -1.000000e+00
  %c2 = fcmp oge float %tmp3, 0.0
  call void @llvm.amdgcn.kill(i1 %c2)
  ret void
}

; SI-LABEL: {{^}}vcc_implicit_def:
; SI-NOT: v_cmp_gt_f32_e32 vcc,
; SI: v_cmp_gt_f32_e64 [[CMP:s\[[0-9]+:[0-9]+\]]], 0, v{{[0-9]+}}
; SI: v_cmpx_le_f32_e32 vcc, 0, v{{[0-9]+}}
; SI: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1.0, [[CMP]]
define amdgpu_ps void @vcc_implicit_def(float %arg13, float %arg14) {
  %tmp0 = fcmp olt float %arg13, 0.000000e+00
  %c1 = fcmp oge float %arg14, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  %tmp1 = select i1 %tmp0, float 1.000000e+00, float 0.000000e+00
  call void @llvm.amdgcn.exp.f32(i32 1, i32 15, float %tmp1, float %tmp1, float %tmp1, float %tmp1, i1 true, i1 true) #0
  ret void
}

; SI-LABEL: {{^}}true:
; SI-NEXT: %bb.
; SI-NEXT: %bb.
; SI-NEXT: s_endpgm
define amdgpu_gs void @true() {
  call void @llvm.amdgcn.kill(i1 true)
  ret void
}

; SI-LABEL: {{^}}false:
; SI-NOT: v_cmpx
; SI: s_mov_b64 exec, 0
define amdgpu_gs void @false() {
  call void @llvm.amdgcn.kill(i1 false)
  ret void
}

; SI-LABEL: {{^}}and:
; SI: v_cmp_lt_i32
; SI: v_cmp_lt_i32
; SI: s_or_b64 s[0:1]
; SI: s_and_b64 exec, exec, s[0:1]
define amdgpu_gs void @and(i32 %a, i32 %b, i32 %c, i32 %d) {
  %c1 = icmp slt i32 %a, %b
  %c2 = icmp slt i32 %c, %d
  %x = or i1 %c1, %c2
  call void @llvm.amdgcn.kill(i1 %x)
  ret void
}

; SI-LABEL: {{^}}andn2:
; SI: v_cmp_lt_i32
; SI: v_cmp_lt_i32
; SI: s_xor_b64 s[0:1]
; SI: s_andn2_b64 exec, exec, s[0:1]
define amdgpu_gs void @andn2(i32 %a, i32 %b, i32 %c, i32 %d) {
  %c1 = icmp slt i32 %a, %b
  %c2 = icmp slt i32 %c, %d
  %x = xor i1 %c1, %c2
  %y = xor i1 %x, 1
  call void @llvm.amdgcn.kill(i1 %y)
  ret void
}

; SI-LABEL: {{^}}oeq:
; SI: v_cmpx_eq_f32
; SI-NOT: s_and
define amdgpu_gs void @oeq(float %a) {
  %c1 = fcmp oeq float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  ret void
}

; SI-LABEL: {{^}}ogt:
; SI: v_cmpx_lt_f32
; SI-NOT: s_and
define amdgpu_gs void @ogt(float %a) {
  %c1 = fcmp ogt float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  ret void
}

; SI-LABEL: {{^}}oge:
; SI: v_cmpx_le_f32
; SI-NOT: s_and
define amdgpu_gs void @oge(float %a) {
  %c1 = fcmp oge float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  ret void
}

; SI-LABEL: {{^}}olt:
; SI: v_cmpx_gt_f32
; SI-NOT: s_and
define amdgpu_gs void @olt(float %a) {
  %c1 = fcmp olt float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  ret void
}

; SI-LABEL: {{^}}ole:
; SI: v_cmpx_ge_f32
; SI-NOT: s_and
define amdgpu_gs void @ole(float %a) {
  %c1 = fcmp ole float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  ret void
}

; SI-LABEL: {{^}}one:
; SI: v_cmpx_lg_f32
; SI-NOT: s_and
define amdgpu_gs void @one(float %a) {
  %c1 = fcmp one float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  ret void
}

; SI-LABEL: {{^}}ord:
; FIXME: This is absolutely unimportant, but we could use the cmpx variant here.
; SI: v_cmp_o_f32
define amdgpu_gs void @ord(float %a) {
  %c1 = fcmp ord float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  ret void
}

; SI-LABEL: {{^}}uno:
; FIXME: This is absolutely unimportant, but we could use the cmpx variant here.
; SI: v_cmp_u_f32
define amdgpu_gs void @uno(float %a) {
  %c1 = fcmp uno float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  ret void
}

; SI-LABEL: {{^}}ueq:
; SI: v_cmpx_nlg_f32
; SI-NOT: s_and
define amdgpu_gs void @ueq(float %a) {
  %c1 = fcmp ueq float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  ret void
}

; SI-LABEL: {{^}}ugt:
; SI: v_cmpx_nge_f32
; SI-NOT: s_and
define amdgpu_gs void @ugt(float %a) {
  %c1 = fcmp ugt float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  ret void
}

; SI-LABEL: {{^}}uge:
; SI: v_cmpx_ngt_f32_e32 vcc, -1.0
; SI-NOT: s_and
define amdgpu_gs void @uge(float %a) {
  %c1 = fcmp uge float %a, -1.0
  call void @llvm.amdgcn.kill(i1 %c1)
  ret void
}

; SI-LABEL: {{^}}ult:
; SI: v_cmpx_nle_f32_e32 vcc, -2.0
; SI-NOT: s_and
define amdgpu_gs void @ult(float %a) {
  %c1 = fcmp ult float %a, -2.0
  call void @llvm.amdgcn.kill(i1 %c1)
  ret void
}

; SI-LABEL: {{^}}ule:
; SI: v_cmpx_nlt_f32_e32 vcc, 2.0
; SI-NOT: s_and
define amdgpu_gs void @ule(float %a) {
  %c1 = fcmp ule float %a, 2.0
  call void @llvm.amdgcn.kill(i1 %c1)
  ret void
}

; SI-LABEL: {{^}}une:
; SI: v_cmpx_neq_f32_e32 vcc, 0
; SI-NOT: s_and
define amdgpu_gs void @une(float %a) {
  %c1 = fcmp une float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  ret void
}

; SI-LABEL: {{^}}neg_olt:
; SI: v_cmpx_ngt_f32_e32 vcc, 1.0
; SI-NOT: s_and
define amdgpu_gs void @neg_olt(float %a) {
  %c1 = fcmp olt float %a, 1.0
  %c2 = xor i1 %c1, 1
  call void @llvm.amdgcn.kill(i1 %c2)
  ret void
}

; SI-LABEL: {{^}}fcmp_x2:
; FIXME: LLVM should be able to combine these fcmp opcodes.
; SI: v_cmp_gt_f32
; SI: v_cndmask_b32
; SI: v_cmpx_le_f32
define amdgpu_ps void @fcmp_x2(float %a) #0 {
  %ogt = fcmp nsz ogt float %a, 2.500000e-01
  %k = select i1 %ogt, float -1.000000e+00, float 0.000000e+00
  %c = fcmp nsz oge float %k, 0.000000e+00
  call void @llvm.amdgcn.kill(i1 %c) #1
  ret void
}

; SI-LABEL: {{^}}wqm:
; SI: v_cmp_neq_f32_e32 vcc, 0
; SI: s_wqm_b64 s[0:1], vcc
; SI: s_and_b64 exec, exec, s[0:1]
define amdgpu_ps void @wqm(float %a) {
  %c1 = fcmp une float %a, 0.0
  %c2 = call i1 @llvm.amdgcn.wqm.vote(i1 %c1)
  call void @llvm.amdgcn.kill(i1 %c2)
  ret void
}

declare void @llvm.amdgcn.kill(i1) #0
declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1) #0
declare i1 @llvm.amdgcn.wqm.vote(i1)

attributes #0 = { nounwind }
