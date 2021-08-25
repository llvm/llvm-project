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
  call void @llvm.amdgcn.s.sendmsg(i32 3, i32 0)
  ret void
}

; GCN-LABEL: {{^}}vcc_implicit_def:
; GCN: v_cmp_nle_f32_e32 vcc, 0, v{{[0-9]+}}
; GCN: v_cmp_gt_f32_e64 [[CMP:s\[[0-9]+:[0-9]+\]]], 0, v{{[0-9]+}}
; GCN: s_andn2_b64 exec, exec, vcc
; GCN: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1.0, [[CMP]]
define amdgpu_ps void @vcc_implicit_def(float %arg13, float %arg14) {
  %tmp0 = fcmp olt float %arg13, 0.000000e+00
  %c1 = fcmp oge float %arg14, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  %tmp1 = select i1 %tmp0, float 1.000000e+00, float 0.000000e+00
  call void @llvm.amdgcn.exp.f32(i32 1, i32 15, float %tmp1, float %tmp1, float %tmp1, float %tmp1, i1 true, i1 true) #0
  call void @llvm.amdgcn.s.sendmsg(i32 3, i32 0)
  ret void
}

; GCN-LABEL: {{^}}true:
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
  call void @llvm.amdgcn.s.sendmsg(i32 3, i32 0)
  ret void
}

; GCN-LABEL: {{^}}and:
; GCN: v_cmp_lt_i32
; GCN: v_cmp_lt_i32
; GCN: s_or_b64 s[0:1]
; GCN: s_xor_b64 s[0:1], s[0:1], exec
; GCN: s_andn2_b64 s[2:3], s[2:3], s[0:1]
; GCN: s_and_b64 exec, exec, s[2:3]
define amdgpu_gs void @and(i32 %a, i32 %b, i32 %c, i32 %d) {
  %c1 = icmp slt i32 %a, %b
  %c2 = icmp slt i32 %c, %d
  %x = or i1 %c1, %c2
  call void @llvm.amdgcn.kill(i1 %x)
  call void @llvm.amdgcn.s.sendmsg(i32 3, i32 0)
  ret void
}

; GCN-LABEL: {{^}}andn2:
; GCN: v_cmp_lt_i32
; GCN: v_cmp_lt_i32
; GCN: s_xor_b64 s[0:1]
; GCN: s_andn2_b64 s[2:3], s[2:3], s[0:1]
; GCN: s_and_b64 exec, exec, s[2:3]
define amdgpu_gs void @andn2(i32 %a, i32 %b, i32 %c, i32 %d) {
  %c1 = icmp slt i32 %a, %b
  %c2 = icmp slt i32 %c, %d
  %x = xor i1 %c1, %c2
  %y = xor i1 %x, 1
  call void @llvm.amdgcn.kill(i1 %y)
  call void @llvm.amdgcn.s.sendmsg(i32 3, i32 0)
  ret void
}

; GCN-LABEL: {{^}}oeq:
; GCN: v_cmp_neq_f32
define amdgpu_gs void @oeq(float %a) {
  %c1 = fcmp oeq float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  call void @llvm.amdgcn.s.sendmsg(i32 3, i32 0)
  ret void
}

; GCN-LABEL: {{^}}ogt:
; GCN: v_cmp_nlt_f32
define amdgpu_gs void @ogt(float %a) {
  %c1 = fcmp ogt float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  call void @llvm.amdgcn.s.sendmsg(i32 3, i32 0)
  ret void
}

; GCN-LABEL: {{^}}oge:
; GCN: v_cmp_nle_f32
define amdgpu_gs void @oge(float %a) {
  %c1 = fcmp oge float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  call void @llvm.amdgcn.s.sendmsg(i32 3, i32 0)
  ret void
}

; GCN-LABEL: {{^}}olt:
; GCN: v_cmp_ngt_f32
define amdgpu_gs void @olt(float %a) {
  %c1 = fcmp olt float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  call void @llvm.amdgcn.s.sendmsg(i32 3, i32 0)
  ret void
}

; GCN-LABEL: {{^}}ole:
; GCN: v_cmp_nge_f32
define amdgpu_gs void @ole(float %a) {
  %c1 = fcmp ole float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  call void @llvm.amdgcn.s.sendmsg(i32 3, i32 0)
  ret void
}

; GCN-LABEL: {{^}}one:
; GCN: v_cmp_nlg_f32
define amdgpu_gs void @one(float %a) {
  %c1 = fcmp one float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  call void @llvm.amdgcn.s.sendmsg(i32 3, i32 0)
  ret void
}

; GCN-LABEL: {{^}}ord:
; GCN: v_cmp_o_f32
define amdgpu_gs void @ord(float %a) {
  %c1 = fcmp ord float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  call void @llvm.amdgcn.s.sendmsg(i32 3, i32 0)
  ret void
}

; GCN-LABEL: {{^}}uno:
; GCN: v_cmp_u_f32
define amdgpu_gs void @uno(float %a) {
  %c1 = fcmp uno float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  call void @llvm.amdgcn.s.sendmsg(i32 3, i32 0)
  ret void
}

; GCN-LABEL: {{^}}ueq:
; GCN: v_cmp_lg_f32
define amdgpu_gs void @ueq(float %a) {
  %c1 = fcmp ueq float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  call void @llvm.amdgcn.s.sendmsg(i32 3, i32 0)
  ret void
}

; GCN-LABEL: {{^}}ugt:
; GCN: v_cmp_ge_f32
define amdgpu_gs void @ugt(float %a) {
  %c1 = fcmp ugt float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  call void @llvm.amdgcn.s.sendmsg(i32 3, i32 0)
  ret void
}

; GCN-LABEL: {{^}}uge:
; GCN: v_cmp_gt_f32_e32 vcc, -1.0
define amdgpu_gs void @uge(float %a) {
  %c1 = fcmp uge float %a, -1.0
  call void @llvm.amdgcn.kill(i1 %c1)
  call void @llvm.amdgcn.s.sendmsg(i32 3, i32 0)
  ret void
}

; GCN-LABEL: {{^}}ult:
; GCN: v_cmp_le_f32_e32 vcc, -2.0
define amdgpu_gs void @ult(float %a) {
  %c1 = fcmp ult float %a, -2.0
  call void @llvm.amdgcn.kill(i1 %c1)
  call void @llvm.amdgcn.s.sendmsg(i32 3, i32 0)
  ret void
}

; GCN-LABEL: {{^}}ule:
; GCN: v_cmp_lt_f32_e32 vcc, 2.0
define amdgpu_gs void @ule(float %a) {
  %c1 = fcmp ule float %a, 2.0
  call void @llvm.amdgcn.kill(i1 %c1)
  call void @llvm.amdgcn.s.sendmsg(i32 3, i32 0)
  ret void
}

; GCN-LABEL: {{^}}une:
; GCN: v_cmp_eq_f32_e32 vcc, 0
define amdgpu_gs void @une(float %a) {
  %c1 = fcmp une float %a, 0.0
  call void @llvm.amdgcn.kill(i1 %c1)
  call void @llvm.amdgcn.s.sendmsg(i32 3, i32 0)
  ret void
}

; GCN-LABEL: {{^}}neg_olt:
; GCN: v_cmp_gt_f32_e32 vcc, 1.0
define amdgpu_gs void @neg_olt(float %a) {
  %c1 = fcmp olt float %a, 1.0
  %c2 = xor i1 %c1, 1
  call void @llvm.amdgcn.kill(i1 %c2)
  call void @llvm.amdgcn.s.sendmsg(i32 3, i32 0)
  ret void
}

; GCN-LABEL: {{^}}fcmp_x2:
; FIXME: LLVM should be able to combine these fcmp opcodes.
; SI: v_cmp_lt_f32_e32 vcc, s{{[0-9]+}}, v0
; GFX10: v_cmp_lt_f32_e32 vcc, 0x3e800000, v0
; GCN: v_cndmask_b32
; GCN: v_cmp_nle_f32
define amdgpu_ps void @fcmp_x2(float %a) #0 {
  %ogt = fcmp nsz ogt float %a, 2.500000e-01
  %k = select i1 %ogt, float -1.000000e+00, float 0.000000e+00
  %c = fcmp nsz oge float %k, 0.000000e+00
  call void @llvm.amdgcn.kill(i1 %c) #1
  ret void
}

; Note: an almost identical test for this exists in llvm.amdgcn.wqm.vote.ll
; GCN-LABEL: {{^}}wqm:
; GCN: v_cmp_neq_f32_e32 vcc, 0
; GCN-DAG: s_wqm_b64 s[2:3], vcc
; GCN-DAG: s_mov_b64 s[0:1], exec
; GCN: s_xor_b64 s[2:3], s[2:3], exec
; GCN: s_andn2_b64 s[0:1], s[0:1], s[2:3]
; GCN: s_and_b64 exec, exec, s[0:1]
define amdgpu_ps float @wqm(float %a) {
  %c1 = fcmp une float %a, 0.0
  %c2 = call i1 @llvm.amdgcn.wqm.vote(i1 %c1)
  call void @llvm.amdgcn.kill(i1 %c2)
  ret float 0.0
}

; This checks that we use the 64-bit encoding when the operand is a SGPR.
; GCN-LABEL: {{^}}test_sgpr:
; GCN: v_cmp_nle_f32_e64
define amdgpu_ps void @test_sgpr(float inreg %a) #0 {
  %c = fcmp ole float %a, 1.000000e+00
  call void @llvm.amdgcn.kill(i1 %c) #1
  ret void
}

; GCN-LABEL: {{^}}test_non_inline_imm_sgpr:
; GCN-NOT: v_cmp_le_f32_e64
define amdgpu_ps void @test_non_inline_imm_sgpr(float inreg %a) #0 {
  %c = fcmp ole float %a, 1.500000e+00
  call void @llvm.amdgcn.kill(i1 %c) #1
  ret void
}

; GCN-LABEL: {{^}}test_scc_liveness:
; GCN: s_cmp
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

; Check this compiles.
; If kill is marked as defining VCC then this will fail with live interval issues.
; GCN-LABEL: {{^}}kill_with_loop_exit:
; GCN: s_mov_b64 [[LIVE:s\[[0-9]+:[0-9]+\]]], exec
; GCN: s_andn2_b64 [[LIVE]], [[LIVE]], exec
; GCN-NEXT: s_cbranch_scc0
define amdgpu_ps void @kill_with_loop_exit(float inreg %inp0, float inreg %inp1, <4 x i32> inreg %inp2, float inreg %inp3) {
.entry:
  %tmp24 = fcmp olt float %inp0, 1.280000e+02
  %tmp25 = fcmp olt float %inp1, 1.280000e+02
  %tmp26 = and i1 %tmp24, %tmp25
  br i1 %tmp26, label %bb35, label %.preheader1.preheader

.preheader1.preheader:                            ; preds = %.entry
  %tmp31 = fcmp ogt float %inp3, 0.0
  br label %bb

bb:                                               ; preds = %bb, %.preheader1.preheader
  %tmp30 = phi float [ %tmp32, %bb ], [ 1.500000e+00, %.preheader1.preheader ]
  %tmp32 = fadd reassoc nnan nsz arcp contract float %tmp30, 2.500000e-01
  %tmp34 = fadd reassoc nnan nsz arcp contract float %tmp30, 2.500000e-01
  br i1 %tmp31, label %bb, label %bb33

bb33:                                             ; preds = %bb
  call void @llvm.amdgcn.kill(i1 false)
  br label %bb35

bb35:                                             ; preds = %bb33, %.entry
  %tmp36 = phi float [ %tmp34, %bb33 ], [ 1.000000e+00, %.entry ]
  call void @llvm.amdgcn.exp.f32(i32 immarg 0, i32 immarg 15, float %tmp36, float %tmp36, float %tmp36, float %tmp36, i1 immarg true, i1 immarg true) #3
  ret void
}

declare void @llvm.amdgcn.kill(i1) #0
declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1) #0
declare void @llvm.amdgcn.s.sendmsg(i32, i32) #0
declare i1 @llvm.amdgcn.wqm.vote(i1)

attributes #0 = { nounwind }
