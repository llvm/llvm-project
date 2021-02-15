; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SI %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -mattr=-wavefrontsize32,+wavefrontsize64 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX10 %s

; GCN-LABEL: {{^}}gs_const:
; GCN-NOT: v_cmpx
; GCN: s_mov_b64 exec, 0
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

; GCN-LABEL: {{^}}vcc_implicit_def:
; GCN-NOT: v_cmp_gt_f32_e32 vcc,
; GCN: v_cmp_gt_f32_e64 [[CMP:s\[[0-9]+:[0-9]+\]]], 0, v{{[0-9]+}}
; SI: v_cmpx_le_f32_e32 vcc, 0, v{{[0-9]+}}
; GFX10: v_cmpx_le_f32_e32 0, v{{[0-9]+}}
; GCN: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1.0, [[CMP]]
define amdgpu_ps void @vcc_implicit_def(float %arg13, float %arg14) {
  %tmp0 = fcmp olt float %arg13, 0.000000e+00
  %c1 = fcmp oge float %arg14, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  %tmp1 = select i1 %tmp0, float 1.000000e+00, float 0.000000e+00
  call void @llvm.amdgcn.exp.f32(i32 1, i32 15, float %tmp1, float %tmp1, float %tmp1, float %tmp1, i1 true, i1 true) #0
  ret void
}

; GCN-LABEL: {{^}}true:
; GCN-NEXT: %bb.
; GCN-NEXT: %bb.
; GCN-NEXT: s_endpgm
define amdgpu_gs void @true() {
  call void @llvm.amdgcn.kill(i1 true)
  ret void
}

; GCN-LABEL: {{^}}false:
; GCN-NOT: v_cmpx
; GCN: s_mov_b64 exec, 0
define amdgpu_gs void @false() {
  call void @llvm.amdgcn.kill(i1 false)
  ret void
}

; GCN-LABEL: {{^}}and:
; GCN: v_cmp_lt_i32
; GCN: v_cmp_lt_i32
; GCN: s_or_b64 s[0:1]
; GCN: s_and_b64 exec, exec, s[0:1]
define amdgpu_gs void @and(i32 %a, i32 %b, i32 %c, i32 %d) {
  %c1 = icmp slt i32 %a, %b
  %c2 = icmp slt i32 %c, %d
  %x = or i1 %c1, %c2
  call void @llvm.amdgcn.kill(i1 %x)
  ret void
}

; GCN-LABEL: {{^}}andn2:
; GCN: v_cmp_lt_i32
; GCN: v_cmp_lt_i32
; GCN: s_xor_b64 s[0:1]
; GCN: s_andn2_b64 exec, exec, s[0:1]
define amdgpu_gs void @andn2(i32 %a, i32 %b, i32 %c, i32 %d) {
  %c1 = icmp slt i32 %a, %b
  %c2 = icmp slt i32 %c, %d
  %x = xor i1 %c1, %c2
  %y = xor i1 %x, 1
  call void @llvm.amdgcn.kill(i1 %y)
  ret void
}

; GCN-LABEL: {{^}}oeq:
; GCN: v_cmpx_eq_f32
; GCN-NOT: s_and
define amdgpu_gs void @oeq(float %a) {
  %c1 = fcmp oeq float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  ret void
}

; GCN-LABEL: {{^}}ogt:
; GCN: v_cmpx_lt_f32
; GCN-NOT: s_and
define amdgpu_gs void @ogt(float %a) {
  %c1 = fcmp ogt float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  ret void
}

; GCN-LABEL: {{^}}oge:
; GCN: v_cmpx_le_f32
; GCN-NOT: s_and
define amdgpu_gs void @oge(float %a) {
  %c1 = fcmp oge float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  ret void
}

; GCN-LABEL: {{^}}olt:
; GCN: v_cmpx_gt_f32
; GCN-NOT: s_and
define amdgpu_gs void @olt(float %a) {
  %c1 = fcmp olt float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  ret void
}

; GCN-LABEL: {{^}}ole:
; GCN: v_cmpx_ge_f32
; GCN-NOT: s_and
define amdgpu_gs void @ole(float %a) {
  %c1 = fcmp ole float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  ret void
}

; GCN-LABEL: {{^}}one:
; GCN: v_cmpx_lg_f32
; GCN-NOT: s_and
define amdgpu_gs void @one(float %a) {
  %c1 = fcmp one float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  ret void
}

; GCN-LABEL: {{^}}ord:
; FIXME: This is absolutely unimportant, but we could use the cmpx variant here.
; GCN: v_cmp_o_f32
define amdgpu_gs void @ord(float %a) {
  %c1 = fcmp ord float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  ret void
}

; GCN-LABEL: {{^}}uno:
; FIXME: This is absolutely unimportant, but we could use the cmpx variant here.
; GCN: v_cmp_u_f32
define amdgpu_gs void @uno(float %a) {
  %c1 = fcmp uno float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  ret void
}

; GCN-LABEL: {{^}}ueq:
; GCN: v_cmpx_nlg_f32
; GCN-NOT: s_and
define amdgpu_gs void @ueq(float %a) {
  %c1 = fcmp ueq float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  ret void
}

; GCN-LABEL: {{^}}ugt:
; GCN: v_cmpx_nge_f32
; GCN-NOT: s_and
define amdgpu_gs void @ugt(float %a) {
  %c1 = fcmp ugt float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  ret void
}

; GCN-LABEL: {{^}}uge:
; SI: v_cmpx_ngt_f32_e32 vcc, -1.0
; GFX10: v_cmpx_ngt_f32_e32 -1.0
; GCN-NOT: s_and
define amdgpu_gs void @uge(float %a) {
  %c1 = fcmp uge float %a, -1.0
  call void @llvm.amdgcn.kill(i1 %c1)
  ret void
}

; GCN-LABEL: {{^}}ult:
; SI: v_cmpx_nle_f32_e32 vcc, -2.0
; GFX10: v_cmpx_nle_f32_e32 -2.0
; GCN-NOT: s_and
define amdgpu_gs void @ult(float %a) {
  %c1 = fcmp ult float %a, -2.0
  call void @llvm.amdgcn.kill(i1 %c1)
  ret void
}

; GCN-LABEL: {{^}}ule:
; SI: v_cmpx_nlt_f32_e32 vcc, 2.0
; GFX10: v_cmpx_nlt_f32_e32 2.0
; GCN-NOT: s_and
define amdgpu_gs void @ule(float %a) {
  %c1 = fcmp ule float %a, 2.0
  call void @llvm.amdgcn.kill(i1 %c1)
  ret void
}

; GCN-LABEL: {{^}}une:
; SI: v_cmpx_neq_f32_e32 vcc, 0
; GFX10: v_cmpx_neq_f32_e32 0
; GCN-NOT: s_and
define amdgpu_gs void @une(float %a) {
  %c1 = fcmp une float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  ret void
}

; GCN-LABEL: {{^}}neg_olt:
; SI: v_cmpx_ngt_f32_e32 vcc, 1.0
; GFX10: v_cmpx_ngt_f32_e32 1.0
; GCN-NOT: s_and
define amdgpu_gs void @neg_olt(float %a) {
  %c1 = fcmp olt float %a, 1.0
  %c2 = xor i1 %c1, 1
  call void @llvm.amdgcn.kill(i1 %c2)
  ret void
}

; GCN-LABEL: {{^}}fcmp_x2:
; FIXME: LLVM should be able to combine these fcmp opcodes.
; SI: v_cmp_lt_f32_e32 vcc, s{{[0-9]+}}, v0
; GFX10: v_cmp_lt_f32_e32 vcc, 0x3e800000, v0
; GCN: v_cndmask_b32
; GCN: v_cmpx_le_f32
define amdgpu_ps void @fcmp_x2(float %a) #0 {
  %ogt = fcmp nsz ogt float %a, 2.500000e-01
  %k = select i1 %ogt, float -1.000000e+00, float 0.000000e+00
  %c = fcmp nsz oge float %k, 0.000000e+00
  call void @llvm.amdgcn.kill(i1 %c) #1
  ret void
}

; GCN-LABEL: {{^}}wqm:
; GCN: v_cmp_neq_f32_e32 vcc, 0
; GCN: s_wqm_b64 s[0:1], vcc
; GCN: s_and_b64 exec, exec, s[0:1]
define amdgpu_ps void @wqm(float %a) {
  %c1 = fcmp une float %a, 0.0
  %c2 = call i1 @llvm.amdgcn.wqm.vote(i1 %c1)
  call void @llvm.amdgcn.kill(i1 %c2)
  ret void
}

; This checks that we use the 64-bit encoding when the operand is a SGPR.
; GCN-LABEL: {{^}}test_sgpr:
; GCN: v_cmpx_ge_f32_e64
define amdgpu_ps void @test_sgpr(float inreg %a) #0 {
  %c = fcmp ole float %a, 1.000000e+00
  call void @llvm.amdgcn.kill(i1 %c) #1
  ret void
}

; GCN-LABEL: {{^}}test_non_inline_imm_sgpr:
; GCN-NOT: v_cmpx_ge_f32_e64
define amdgpu_ps void @test_non_inline_imm_sgpr(float inreg %a) #0 {
  %c = fcmp ole float %a, 1.500000e+00
  call void @llvm.amdgcn.kill(i1 %c) #1
  ret void
}

; GCN-LABEL: {{^}}test_scc_liveness:
; GCN: v_cmp
; GCN: s_and_b64 exec
; GCN: s_cmp
; GCN: s_cbranch_scc
define amdgpu_ps void @test_scc_liveness() #0 {
main_body:
  br label %loop3

loop3:                                            ; preds = %loop3, %main_body
  %tmp = phi i32 [ 0, %main_body ], [ %tmp5, %loop3 ]
  %tmp1 = icmp sgt i32 %tmp, 0
  call void @llvm.amdgcn.kill(i1 %tmp1) #1
  %tmp5 = add i32 %tmp, 1
  br i1 %tmp1, label %endloop15, label %loop3

endloop15:                                        ; preds = %loop3
  ret void
}

declare void @llvm.amdgcn.kill(i1) #0
declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1) #0
declare i1 @llvm.amdgcn.wqm.vote(i1)

attributes #0 = { nounwind }
