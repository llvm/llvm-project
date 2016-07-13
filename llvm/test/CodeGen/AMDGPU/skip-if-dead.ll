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
; CHECK: s_cmp_lg_i32 s{{[0-9]+}}, 0
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
; CHECK-NEXT: ; BB#3:
; CHECK-NEXT: exp 0, 9, 0, 1, 1, v0, v0, v0, v0
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
; CHECK: s_cmp_lg_i32 s{{[0-9]+}}, 0
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

; CHECK-NEXT: ; BB#4:
; CHECK-NEXT: exp 0, 9, 0, 1, 1, v0, v0, v0, v0
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
; CHECK: v_cmp_eq_i32_e32 vcc, 0, v0
; CHECK-NEXT: s_and_saveexec_b64 [[SAVEEXEC:s\[[0-9]+:[0-9]+\]]], vcc
; CHECK-NEXT: s_xor_b64 [[SAVEEXEC]], exec, [[SAVEEXEC]]
; CHECK-NEXT: s_cbranch_execz [[EXIT:BB[0-9]+_[0-9]+]]
; CHECK-NEXT: ; mask branch [[EXIT]]

; CHECK: [[LOOP_BB:BB[0-9]+_[0-9]+]]:

; CHECK: v_mov_b32_e64 v7, -1
; CHECK: v_nop_e64
; CHECK: v_cmpx_le_f32_e32 vcc, 0, v7

; CHECK-NEXT: ; BB#3:
; CHECK: buffer_load_dword [[LOAD:v[0-9]+]]
; CHECK: v_cmp_eq_i32_e32 vcc, 0, [[LOAD]]
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


declare void @llvm.AMDGPU.kill(float) #0

attributes #0 = { nounwind }
