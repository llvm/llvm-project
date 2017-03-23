; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: {{^}}test_kill_depth_0_imm_pos:
; CHECK-NEXT: ; BB#0:
; CHECK-NEXT: s_endpgm
define amdgpu_ps void @test_kill_depth_0_imm_pos() #0 {
  call void @llvm.AMDGPU.kill(float 0.0)
  ret void
}

; CHECK-LABEL: {{^}}test_kill_depth_0_imm_neg:
; CHECK-NEXT: ; BB#0:
; CHECK-NEXT: s_mov_b64 exec, 0
; CHECK-NEXT: ; BB#1:
; CHECK-NEXT: s_endpgm
define amdgpu_ps void @test_kill_depth_0_imm_neg() #0 {
  call void @llvm.AMDGPU.kill(float -0.0)
  ret void
}

; FIXME: Ideally only one would be emitted
; CHECK-LABEL: {{^}}test_kill_depth_0_imm_neg_x2:
; CHECK-NEXT: ; BB#0:
; CHECK-NEXT: s_mov_b64 exec, 0
; CHECK-NEXT: ; BB#1:
; CHECK-NEXT: s_mov_b64 exec, 0
; CHECK-NEXT: ; BB#2:
; CHECK-NEXT: s_endpgm
define amdgpu_ps void @test_kill_depth_0_imm_neg_x2() #0 {
  call void @llvm.AMDGPU.kill(float -0.0)
  call void @llvm.AMDGPU.kill(float -1.0)
  ret void
}

; CHECK-LABEL: {{^}}test_kill_depth_var:
; CHECK-NEXT: ; BB#0:
; CHECK-NEXT: v_cmpx_le_f32_e32 vcc, 0, v0
; CHECK-NEXT: ; BB#1:
; CHECK-NEXT: s_endpgm
define amdgpu_ps void @test_kill_depth_var(float %x) #0 {
  call void @llvm.AMDGPU.kill(float %x)
  ret void
}

; FIXME: Ideally only one would be emitted
; CHECK-LABEL: {{^}}test_kill_depth_var_x2_same:
; CHECK-NEXT: ; BB#0:
; CHECK-NEXT: v_cmpx_le_f32_e32 vcc, 0, v0
; CHECK-NEXT: ; BB#1:
; CHECK-NEXT: v_cmpx_le_f32_e32 vcc, 0, v0
; CHECK-NEXT: ; BB#2:
; CHECK-NEXT: s_endpgm
define amdgpu_ps void @test_kill_depth_var_x2_same(float %x) #0 {
  call void @llvm.AMDGPU.kill(float %x)
  call void @llvm.AMDGPU.kill(float %x)
  ret void
}

; CHECK-LABEL: {{^}}test_kill_depth_var_x2:
; CHECK-NEXT: ; BB#0:
; CHECK-NEXT: v_cmpx_le_f32_e32 vcc, 0, v0
; CHECK-NEXT: ; BB#1:
; CHECK-NEXT: v_cmpx_le_f32_e32 vcc, 0, v1
; CHECK-NEXT: ; BB#2:
; CHECK-NEXT: s_endpgm
define amdgpu_ps void @test_kill_depth_var_x2(float %x, float %y) #0 {
  call void @llvm.AMDGPU.kill(float %x)
  call void @llvm.AMDGPU.kill(float %y)
  ret void
}

; CHECK-LABEL: {{^}}test_kill_depth_var_x2_instructions:
; CHECK-NEXT: ; BB#0:
; CHECK-NEXT: v_cmpx_le_f32_e32 vcc, 0, v0
; CHECK-NEXT: ; BB#1:
; CHECK: v_mov_b32_e64 v7, -1
; CHECK: v_cmpx_le_f32_e32 vcc, 0, v7
; CHECK-NEXT: ; BB#2:
; CHECK-NEXT: s_endpgm
define amdgpu_ps void @test_kill_depth_var_x2_instructions(float %x) #0 {
  call void @llvm.AMDGPU.kill(float %x)
  %y = call float asm sideeffect "v_mov_b32_e64 v7, -1", "={VGPR7}"()
  call void @llvm.AMDGPU.kill(float %y)
  ret void
}

; FIXME: why does the skip depend on the asm length in the same block?

; CHECK-LABEL: {{^}}test_kill_control_flow:
; CHECK: s_cmp_lg_u32 s{{[0-9]+}}, 0
; CHECK: s_cbranch_scc1 [[RETURN_BB:BB[0-9]+_[0-9]+]]

; CHECK-NEXT: ; BB#1:
; CHECK: v_mov_b32_e64 v7, -1
; CHECK: v_nop_e64
; CHECK: v_nop_e64
; CHECK: v_nop_e64
; CHECK: v_nop_e64
; CHECK: v_nop_e64
; CHECK: v_nop_e64
; CHECK: v_nop_e64
; CHECK: v_nop_e64
; CHECK: v_nop_e64
; CHECK: v_nop_e64

; CHECK: v_cmpx_le_f32_e32 vcc, 0, v7
; CHECK-NEXT: s_cbranch_execnz [[SPLIT_BB:BB[0-9]+_[0-9]+]]
; CHECK-NEXT: ; BB#2:
; CHECK-NEXT: exp null off, off, off, off done vm
; CHECK-NEXT: s_endpgm

; CHECK-NEXT: {{^}}[[SPLIT_BB]]:
; CHECK-NEXT: s_endpgm
define amdgpu_ps void @test_kill_control_flow(i32 inreg %arg) #0 {
entry:
  %cmp = icmp eq i32 %arg, 0
  br i1 %cmp, label %bb, label %exit

bb:
  %var = call float asm sideeffect "
    v_mov_b32_e64 v7, -1
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64", "={VGPR7}"()
  call void @llvm.AMDGPU.kill(float %var)
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: {{^}}test_kill_control_flow_remainder:
; CHECK: s_cmp_lg_u32 s{{[0-9]+}}, 0
; CHECK-NEXT: v_mov_b32_e32 v{{[0-9]+}}, 0
; CHECK-NEXT: s_cbranch_scc1 [[RETURN_BB:BB[0-9]+_[0-9]+]]

; CHECK-NEXT: ; BB#1: ; %bb
; CHECK: v_mov_b32_e64 v7, -1
; CHECK: v_nop_e64
; CHECK: v_nop_e64
; CHECK: v_nop_e64
; CHECK: v_nop_e64
; CHECK: v_nop_e64
; CHECK: v_nop_e64
; CHECK: v_nop_e64
; CHECK: v_nop_e64
; CHECK: ;;#ASMEND
; CHECK: v_mov_b32_e64 v8, -1
; CHECK: ;;#ASMEND
; CHECK: v_cmpx_le_f32_e32 vcc, 0, v7
; CHECK-NEXT: s_cbranch_execnz [[SPLIT_BB:BB[0-9]+_[0-9]+]]

; CHECK-NEXT: ; BB#2:
; CHECK-NEXT: exp null off, off, off, off done vm
; CHECK-NEXT: s_endpgm

; CHECK-NEXT: {{^}}[[SPLIT_BB]]:
; CHECK: buffer_store_dword v8
; CHECK: v_mov_b32_e64 v9, -2

; CHECK: {{^}}BB{{[0-9]+_[0-9]+}}:
; CHECK: buffer_store_dword v9
; CHECK-NEXT: s_endpgm
define amdgpu_ps void @test_kill_control_flow_remainder(i32 inreg %arg) #0 {
entry:
  %cmp = icmp eq i32 %arg, 0
  br i1 %cmp, label %bb, label %exit

bb:
  %var = call float asm sideeffect "
    v_mov_b32_e64 v7, -1
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64", "={VGPR7}"()
  %live.across = call float asm sideeffect "v_mov_b32_e64 v8, -1", "={VGPR8}"()
  call void @llvm.AMDGPU.kill(float %var)
  store volatile float %live.across, float addrspace(1)* undef
  %live.out = call float asm sideeffect "v_mov_b32_e64 v9, -2", "={VGPR9}"()
  br label %exit

exit:
  %phi = phi float [ 0.0, %entry ], [ %live.out, %bb ]
  store float %phi, float addrspace(1)* undef
  ret void
}

; CHECK-LABEL: {{^}}test_kill_divergent_loop:
; CHECK: v_cmp_eq_u32_e32 vcc, 0, v0
; CHECK-NEXT: s_and_saveexec_b64 [[SAVEEXEC:s\[[0-9]+:[0-9]+\]]], vcc
; CHECK-NEXT: s_xor_b64 [[SAVEEXEC]], exec, [[SAVEEXEC]]
; CHECK-NEXT: ; mask branch [[EXIT:BB[0-9]+_[0-9]+]]
; CHECK-NEXT: s_cbranch_execz [[EXIT]]

; CHECK: {{BB[0-9]+_[0-9]+}}: ; %bb.preheader
; CHECK: s_mov_b32

; CHECK: [[LOOP_BB:BB[0-9]+_[0-9]+]]:

; CHECK: v_mov_b32_e64 v7, -1
; CHECK: v_nop_e64
; CHECK: v_cmpx_le_f32_e32 vcc, 0, v7

; CHECK-NEXT: ; BB#3:
; CHECK: buffer_load_dword [[LOAD:v[0-9]+]]
; CHECK: v_cmp_eq_u32_e32 vcc, 0, [[LOAD]]
; CHECK-NEXT: s_and_b64 vcc, exec, vcc
; CHECK-NEXT: s_cbranch_vccnz [[LOOP_BB]]

; CHECK-NEXT: {{^}}[[EXIT]]:
; CHECK: s_or_b64 exec, exec, [[SAVEEXEC]]
; CHECK: buffer_store_dword
; CHECK: s_endpgm
define amdgpu_ps void @test_kill_divergent_loop(i32 %arg) #0 {
entry:
  %cmp = icmp eq i32 %arg, 0
  br i1 %cmp, label %bb, label %exit

bb:
  %var = call float asm sideeffect "
    v_mov_b32_e64 v7, -1
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64
    v_nop_e64", "={VGPR7}"()
  call void @llvm.AMDGPU.kill(float %var)
  %vgpr = load volatile i32, i32 addrspace(1)* undef
  %loop.cond = icmp eq i32 %vgpr, 0
  br i1 %loop.cond, label %bb, label %exit

exit:
  store volatile i32 8, i32 addrspace(1)* undef
  ret void
}

; bug 28550
; CHECK-LABEL: {{^}}phi_use_def_before_kill:
; CHECK: v_cndmask_b32_e64 [[PHIREG:v[0-9]+]], 0, -1.0,
; CHECK: v_cmpx_le_f32_e32 vcc, 0,
; CHECK-NEXT: s_cbranch_execnz [[BB4:BB[0-9]+_[0-9]+]]

; CHECK: exp
; CHECK-NEXT: s_endpgm

; CHECK: [[KILLBB:BB[0-9]+_[0-9]+]]:
; CHECK-NEXT: s_cbranch_scc0 [[PHIBB:BB[0-9]+_[0-9]+]]

; CHECK: [[PHIBB]]:
; CHECK: v_cmp_eq_f32_e32 vcc, 0, [[PHIREG]]
; CHECK-NEXT: s_cbranch_vccz [[ENDBB:BB[0-9]+_[0-9]+]]

; CHECK: ; %bb10
; CHECK: v_mov_b32_e32 v{{[0-9]+}}, 9
; CHECK: buffer_store_dword

; CHECK: [[ENDBB]]:
; CHECK-NEXT: s_endpgm
define amdgpu_ps void @phi_use_def_before_kill() #0 {
bb:
  %tmp = fadd float undef, 1.000000e+00
  %tmp1 = fcmp olt float 0.000000e+00, %tmp
  %tmp2 = select i1 %tmp1, float -1.000000e+00, float 0.000000e+00
  call void @llvm.AMDGPU.kill(float %tmp2)
  br i1 undef, label %phibb, label %bb8

phibb:
  %tmp5 = phi float [ %tmp2, %bb ], [ 4.0, %bb8 ]
  %tmp6 = fcmp oeq float %tmp5, 0.000000e+00
  br i1 %tmp6, label %bb10, label %end

bb8:
  store volatile i32 8, i32 addrspace(1)* undef
  br label %phibb

bb10:
  store volatile i32 9, i32 addrspace(1)* undef
  br label %end

end:
  ret void
}

; CHECK-LABEL: {{^}}no_skip_no_successors:
; CHECK: v_cmp_nge_f32
; CHECK-NEXT: s_cbranch_vccz [[SKIPKILL:BB[0-9]+_[0-9]+]]

; CHECK: ; %bb6
; CHECK: s_mov_b64 exec, 0

; CHECK: [[SKIPKILL]]:
; CHECK: v_cmp_nge_f32_e32 vcc
; CHECK-NEXT: BB#3: ; %bb5
; CHECK-NEXT: .Lfunc_end{{[0-9]+}}
define amdgpu_ps void @no_skip_no_successors(float inreg %arg, float inreg %arg1) #0 {
bb:
  %tmp = fcmp ult float %arg1, 0.000000e+00
  %tmp2 = fcmp ult float %arg, 0x3FCF5C2900000000
  br i1 %tmp, label %bb6, label %bb3

bb3:                                              ; preds = %bb
  br i1 %tmp2, label %bb5, label %bb4

bb4:                                              ; preds = %bb3
  br i1 true, label %bb5, label %bb7

bb5:                                              ; preds = %bb4, %bb3
  unreachable

bb6:                                              ; preds = %bb
  call void @llvm.AMDGPU.kill(float -1.000000e+00)
  unreachable

bb7:                                              ; preds = %bb4
  ret void
}

; CHECK-LABEL: {{^}}if_after_kill_block:
; CHECK: ; BB#0:
; CHECK: s_and_saveexec_b64
; CHECK: s_xor_b64
; CHECK-NEXT: mask branch [[BB4:BB[0-9]+_[0-9]+]]

; CHECK: v_cmpx_le_f32_e32 vcc, 0,
; CHECK: [[BB4]]:
; CHECK: s_or_b64 exec, exec
; CHECK: image_sample_c

; CHECK: v_cmp_neq_f32_e32 vcc, 0,
; CHECK: s_and_saveexec_b64 s{{\[[0-9]+:[0-9]+\]}}, vcc
; CHECK: s_xor_b64 s{{\[[0-9]+:[0-9]+\]}}, exec
; CHECK: mask branch [[END:BB[0-9]+_[0-9]+]]
; CHECK-NOT: branch

; CHECK: BB{{[0-9]+_[0-9]+}}: ; %bb8
; CHECK: buffer_store_dword

; CHECK: [[END]]:
; CHECK: s_or_b64 exec, exec
; CHECK: s_endpgm
define amdgpu_ps void @if_after_kill_block(float %arg, float %arg1, <4 x float> %arg2) #0 {
bb:
  %tmp = fcmp ult float %arg1, 0.000000e+00
  br i1 %tmp, label %bb3, label %bb4

bb3:                                              ; preds = %bb
  call void @llvm.AMDGPU.kill(float %arg)
  br label %bb4

bb4:                                              ; preds = %bb3, %bb
  %tmp5 = call <4 x float> @llvm.amdgcn.image.sample.c.v4f32.v4f32.v8i32(<4 x float> %arg2, <8 x i32> undef, <4 x i32> undef, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %tmp6 = extractelement <4 x float> %tmp5, i32 0
  %tmp7 = fcmp une float %tmp6, 0.000000e+00
  br i1 %tmp7, label %bb8, label %bb9

bb8:                                              ; preds = %bb9, %bb4
  store volatile i32 9, i32 addrspace(1)* undef
  ret void

bb9:                                              ; preds = %bb4
  ret void
}

declare <4 x float> @llvm.amdgcn.image.sample.c.v4f32.v4f32.v8i32(<4 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #1
declare void @llvm.AMDGPU.kill(float) #0

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
