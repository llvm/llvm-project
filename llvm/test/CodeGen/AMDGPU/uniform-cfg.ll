; RUN: llc -march=amdgcn -mcpu=verde -amdgpu-early-ifcvt=0 -machine-sink-split-probability-threshold=0 -structurizecfg-skip-uniform-regions -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -amdgpu-early-ifcvt=0 -machine-sink-split-probability-threshold=0 -structurizecfg-skip-uniform-regions -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=VI %s

; GCN-LABEL: {{^}}uniform_if_scc:
; GCN-DAG: s_cmp_eq_u32 s{{[0-9]+}}, 0
; GCN-DAG: s_mov_b32 [[S_VAL:s[0-9]+]], 0
; GCN: s_cbranch_scc1 [[IF_LABEL:.L[0-9_A-Za-z]+]]

; Fall-through to the else
; GCN: s_mov_b32 [[S_VAL]], 1

; GCN: [[IF_LABEL]]:
; GCN: v_mov_b32_e32 [[V_VAL:v[0-9]+]], [[S_VAL]]
; GCN: buffer_store_dword [[V_VAL]]
define amdgpu_kernel void @uniform_if_scc(i32 %cond, i32 addrspace(1)* %out) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %else

if:
  br label %done

else:
  br label %done

done:
  %value = phi i32 [0, %if], [1, %else]
  store i32 %value, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}uniform_if_vcc:
; GCN-DAG: v_cmp_eq_f32_e64 [[COND:vcc|s\[[0-9]+:[0-9]+\]]], s{{[0-9]+}}, 0{{$}}
; GCN-DAG: s_mov_b32 [[S_VAL:s[0-9]+]], 0
; GCN: s_cbranch_vccnz [[IF_LABEL:.L[0-9_A-Za-z]+]]

; Fall-through to the else
; GCN: s_mov_b32 [[S_VAL]], 1

; GCN: [[IF_LABEL]]:
; GCN: v_mov_b32_e32 [[V_VAL:v[0-9]+]], [[S_VAL]]
; GCN: buffer_store_dword [[V_VAL]]
define amdgpu_kernel void @uniform_if_vcc(float %cond, i32 addrspace(1)* %out) {
entry:
  %cmp0 = fcmp oeq float %cond, 0.0
  br i1 %cmp0, label %if, label %else

if:
  br label %done

else:
  br label %done

done:
  %value = phi i32 [0, %if], [1, %else]
  store i32 %value, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}uniform_if_swap_br_targets_scc:
; GCN-DAG: s_cmp_lg_u32 s{{[0-9]+}}, 0
; GCN-DAG: s_mov_b32 [[S_VAL:s[0-9]+]], 0
; GCN: s_cbranch_scc1 [[IF_LABEL:.L[0-9_A-Za-z]+]]

; Fall-through to the else
; GCN: s_mov_b32 [[S_VAL]], 1

; GCN: [[IF_LABEL]]:
; GCN: v_mov_b32_e32 [[V_VAL:v[0-9]+]], [[S_VAL]]
; GCN: buffer_store_dword [[V_VAL]]
define amdgpu_kernel void @uniform_if_swap_br_targets_scc(i32 %cond, i32 addrspace(1)* %out) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %else, label %if

if:
  br label %done

else:
  br label %done

done:
  %value = phi i32 [0, %if], [1, %else]
  store i32 %value, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}uniform_if_swap_br_targets_vcc:
; GCN-DAG: v_cmp_neq_f32_e64 [[COND:vcc|s\[[0-9]+:[0-9]+\]]], s{{[0-9]+}}, 0{{$}}
; GCN-DAG: s_mov_b32 [[S_VAL:s[0-9]+]], 0
; GCN: s_cbranch_vccnz [[IF_LABEL:.L[0-9_A-Za-z]+]]

; Fall-through to the else
; GCN: s_mov_b32 [[S_VAL]], 1

; GCN: [[IF_LABEL]]:
; GCN: v_mov_b32_e32 [[V_VAL:v[0-9]+]], [[S_VAL]]
; GCN: buffer_store_dword [[V_VAL]]
define amdgpu_kernel void @uniform_if_swap_br_targets_vcc(float %cond, i32 addrspace(1)* %out) {
entry:
  %cmp0 = fcmp oeq float %cond, 0.0
  br i1 %cmp0, label %else, label %if

if:
  br label %done

else:
  br label %done

done:
  %value = phi i32 [0, %if], [1, %else]
  store i32 %value, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}uniform_if_move_valu:
; GCN: v_add_f32_e32 [[CMP:v[0-9]+]]
; Using a floating-point value in an integer compare will cause the compare to
; be selected for the SALU and then later moved to the VALU.
; GCN: v_cmp_ne_u32_e32 [[COND:vcc|s\[[0-9]+:[0-9]+\]]], 5, [[CMP]]
; GCN: s_cbranch_vccnz [[ENDIF_LABEL:.L[0-9_A-Za-z]+]]
; GCN: buffer_store_dword
; GCN: [[ENDIF_LABEL]]:
; GCN: s_endpgm
define amdgpu_kernel void @uniform_if_move_valu(i32 addrspace(1)* %out, float %a) {
entry:
  %a.0 = fadd float %a, 10.0
  %cond = bitcast float %a.0 to i32
  %cmp = icmp eq i32 %cond, 5
  br i1 %cmp, label %if, label %endif

if:
  store i32 0, i32 addrspace(1)* %out
  br label %endif

endif:
  ret void
}

; GCN-LABEL: {{^}}uniform_if_move_valu_commute:
; GCN: v_add_f32_e32 [[CMP:v[0-9]+]]
; Using a floating-point value in an integer compare will cause the compare to
; be selected for the SALU and then later moved to the VALU.
; GCN: v_cmp_gt_u32_e32 [[COND:vcc|s\[[0-9]+:[0-9]+\]]], 6, [[CMP]]
; GCN: s_cbranch_vccnz [[ENDIF_LABEL:.L[0-9_A-Za-z]+]]
; GCN: buffer_store_dword
; GCN: [[ENDIF_LABEL]]:
; GCN: s_endpgm
define amdgpu_kernel void @uniform_if_move_valu_commute(i32 addrspace(1)* %out, float %a) {
entry:
  %a.0 = fadd float %a, 10.0
  %cond = bitcast float %a.0 to i32
  %cmp = icmp ugt i32 %cond, 5
  br i1 %cmp, label %if, label %endif

if:
  store i32 0, i32 addrspace(1)* %out
  br label %endif

endif:
  ret void
}


; GCN-LABEL: {{^}}uniform_if_else_ret:
; GCN: s_cmp_lg_u32 s{{[0-9]+}}, 0
; GCN: s_cbranch_scc0 [[IF_LABEL:.L[0-9_A-Za-z]+]]

; GCN: v_mov_b32_e32 [[TWO:v[0-9]+]], 2
; GCN: buffer_store_dword [[TWO]]
; GCN: s_endpgm

; GCN: {{^}}[[IF_LABEL]]:
; GCN: v_mov_b32_e32 [[ONE:v[0-9]+]], 1
; GCN: buffer_store_dword [[ONE]]
; GCN: s_endpgm
define amdgpu_kernel void @uniform_if_else_ret(i32 addrspace(1)* nocapture %out, i32 %a) {
entry:
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  store i32 1, i32 addrspace(1)* %out
  br label %if.end

if.else:                                          ; preds = %entry
  store i32 2, i32 addrspace(1)* %out
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

; GCN-LABEL: {{^}}uniform_if_else:
; GCN: s_cmp_lg_u32 s{{[0-9]+}}, 0
; GCN: s_cbranch_scc0 [[IF_LABEL:.L[0-9_A-Za-z]+]]

; GCN: v_mov_b32_e32 [[IMM_REG:v[0-9]+]], 2
; GCN: s_branch [[ENDIF_LABEL:.L[0-9_A-Za-z]+]]

; GCN: [[IF_LABEL]]:
; GCN: v_mov_b32_e32 [[IMM_REG]], 1

; GCN-NEXT: [[ENDIF_LABEL]]:
; GCN: buffer_store_dword [[IMM_REG]]

; GCN: v_mov_b32_e32 [[THREE:v[0-9]+]], 3
; GCN: buffer_store_dword [[THREE]]
; GCN: s_endpgm
define amdgpu_kernel void @uniform_if_else(i32 addrspace(1)* nocapture %out0, i32 addrspace(1)* nocapture %out1, i32 %a) {
entry:
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  store i32 1, i32 addrspace(1)* %out0
  br label %if.end

if.else:                                          ; preds = %entry
  store i32 2, i32 addrspace(1)* %out0
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  store i32 3, i32 addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}icmp_2_users:
; GCN: s_cmp_lt_i32 s{{[0-9]+}}, 1
; GCN: s_cbranch_scc1 [[LABEL:.L[0-9_A-Za-z]+]]
; GCN: buffer_store_dword
; GCN: [[LABEL]]:
; GCN: s_endpgm
define amdgpu_kernel void @icmp_2_users(i32 addrspace(1)* %out, i32 %cond) {
main_body:
  %0 = icmp sgt i32 %cond, 0
  %1 = sext i1 %0 to i32
  br i1 %0, label %IF, label %ENDIF

IF:
  store i32 %1, i32 addrspace(1)* %out
  br label %ENDIF

ENDIF:                                            ; preds = %IF, %main_body
  ret void
}

; GCN-LABEL: {{^}}icmp_users_different_blocks:
; GCN: s_load_dwordx2 s[[[COND0:[0-9]+]]:[[COND1:[0-9]+]]]
; GCN: s_cmp_lt_i32 s[[COND0]], 1
; GCN: s_cbranch_scc1 [[EXIT:.L[0-9_A-Za-z]+]]
; GCN: s_cmp_gt_i32 s[[COND1]], 0{{$}}
; GCN: s_cbranch_vccz [[BODY:.L[0-9_A-Za-z]+]]
; GCN: {{^}}[[EXIT]]:
; GCN: s_endpgm
; GCN: {{^}}[[BODY]]:
; GCN: buffer_store
; GCN: s_endpgm
define amdgpu_kernel void @icmp_users_different_blocks(i32 %cond0, i32 %cond1, i32 addrspace(1)* %out) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %cmp0 = icmp sgt i32 %cond0, 0
  %cmp1 = icmp sgt i32 %cond1, 0
  br i1 %cmp0, label %bb2, label %bb9

bb2:                                              ; preds = %bb
  %tmp2 = sext i1 %cmp1 to i32
  %tmp3 = add i32 %tmp2, %tmp
  br i1 %cmp1, label %bb9, label %bb7

bb7:                                              ; preds = %bb5
  store i32 %tmp3, i32 addrspace(1)* %out
  br label %bb9

bb9:                                              ; preds = %bb8, %bb4
  ret void
}

; SI-LABEL: {{^}}uniform_loop:
; SI: {{^}}[[LOOP_LABEL:.L[0-9_A-Za-z]+]]:
; SI: s_add_i32 [[I:s[0-9]+]],  s{{[0-9]+}}, -1
; SI: s_cmp_lg_u32 [[I]], 0
; SI: s_cbranch_scc1 [[LOOP_LABEL]]
; SI: s_endpgm
define amdgpu_kernel void @uniform_loop(i32 addrspace(1)* %out, i32 %a) {
entry:
  br label %loop

loop:
  %i = phi i32 [0, %entry], [%i.i, %loop]
  %i.i = add i32 %i, 1
  %cmp = icmp eq i32 %a, %i.i
  br i1 %cmp, label %done, label %loop

done:
  ret void
}

; Test uniform and divergent.

; GCN-LABEL: {{^}}uniform_inside_divergent:
; GCN: v_cmp_gt_u32_e32 vcc, 16, v{{[0-9]+}}
; GCN: s_and_saveexec_b64 [[MASK:s\[[0-9]+:[0-9]+\]]], vcc
; GCN: s_cmp_lg_u32 {{s[0-9]+}}, 0
; GCN: s_cbranch_scc0 [[IF_UNIFORM_LABEL:.L[0-9_A-Za-z]+]]
; GCN: s_endpgm
; GCN: {{^}}[[IF_UNIFORM_LABEL]]:
; GCN: v_mov_b32_e32 [[ONE:v[0-9]+]], 1
; GCN: buffer_store_dword [[ONE]]
define amdgpu_kernel void @uniform_inside_divergent(i32 addrspace(1)* %out, i32 %cond) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %d_cmp = icmp ult i32 %tid, 16
  br i1 %d_cmp, label %if, label %endif

if:
  store i32 0, i32 addrspace(1)* %out
  %u_cmp = icmp eq i32 %cond, 0
  br i1 %u_cmp, label %if_uniform, label %endif

if_uniform:
  store i32 1, i32 addrspace(1)* %out
  br label %endif

endif:
  ret void
}

; GCN-LABEL: {{^}}divergent_inside_uniform:
; GCN: s_cmp_lg_u32 s{{[0-9]+}}, 0
; GCN: s_cbranch_scc0 [[IF_LABEL:.L[0-9_A-Za-z]+]]
; GCN: [[ENDIF_LABEL:.L[0-9_A-Za-z]+]]:
; GCN: [[IF_LABEL]]:
; GCN: v_cmp_gt_u32_e32 vcc, 16, v{{[0-9]+}}
; GCN: s_and_saveexec_b64 [[MASK:s\[[0-9]+:[0-9]+\]]], vcc
; GCN: s_cbranch_execz [[ENDIF_LABEL]]
; GCN: v_mov_b32_e32 [[ONE:v[0-9]+]], 1
; GCN: buffer_store_dword [[ONE]]
; GCN: s_endpgm
define amdgpu_kernel void @divergent_inside_uniform(i32 addrspace(1)* %out, i32 %cond) {
entry:
  %u_cmp = icmp eq i32 %cond, 0
  br i1 %u_cmp, label %if, label %endif

if:
  store i32 0, i32 addrspace(1)* %out
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %d_cmp = icmp ult i32 %tid, 16
  br i1 %d_cmp, label %if_uniform, label %endif

if_uniform:
  store i32 1, i32 addrspace(1)* %out
  br label %endif

endif:
  ret void
}

; GCN-LABEL: {{^}}divergent_if_uniform_if:
; GCN: v_cmp_eq_u32_e32 vcc, 0, v0
; GCN: s_and_saveexec_b64 [[MASK:s\[[0-9]+:[0-9]+\]]], vcc
; GCN: v_mov_b32_e32 [[ONE:v[0-9]+]], 1
; GCN: buffer_store_dword [[ONE]]
; GCN: s_or_b64 exec, exec, [[MASK]]
; GCN: s_cmp_lg_u32 s{{[0-9]+}}, 0
; GCN: s_cbranch_scc0 [[IF_UNIFORM:.L[0-9_A-Za-z]+]]
; GCN: s_endpgm
; GCN: [[IF_UNIFORM]]:
; GCN: v_mov_b32_e32 [[TWO:v[0-9]+]], 2
; GCN: buffer_store_dword [[TWO]]
define amdgpu_kernel void @divergent_if_uniform_if(i32 addrspace(1)* %out, i32 %cond) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #0
  %d_cmp = icmp eq i32 %tid, 0
  br i1 %d_cmp, label %if, label %endif

if:
  store i32 1, i32 addrspace(1)* %out
  br label %endif

endif:
  %u_cmp = icmp eq i32 %cond, 0
  br i1 %u_cmp, label %if_uniform, label %exit

if_uniform:
  store i32 2, i32 addrspace(1)* %out
  br label %exit

exit:
  ret void
}

; The condition of the branches in the two blocks are
; uniform. MachineCSE replaces the 2nd condition with the inverse of
; the first, leaving an scc use in a different block than it was
; defed.

; GCN-LABEL: {{^}}cse_uniform_condition_different_blocks:
; GCN: s_load_dword [[COND:s[0-9]+]]
; GCN: s_cmp_lt_i32 [[COND]], 1
; GCN: s_cbranch_scc1 .LBB[[FNNUM:[0-9]+]]_3

; GCN: %bb.1:
; GCN-NOT: cmp
; GCN: buffer_load_dword
; GCN: buffer_store_dword
; GCN: s_cbranch_scc1 .LBB[[FNNUM]]_3

; GCN: .LBB[[FNNUM]]_3:
; GCN: s_endpgm
define amdgpu_kernel void @cse_uniform_condition_different_blocks(i32 %cond, i32 addrspace(1)* %out) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %tmp1 = icmp sgt i32 %cond, 0
  br i1 %tmp1, label %bb2, label %bb9

bb2:                                              ; preds = %bb
  %tmp3 = load volatile i32, i32 addrspace(1)* undef
  store volatile i32 0, i32 addrspace(1)* undef
  %tmp9 = icmp sle i32 %cond, 0
  br i1 %tmp9, label %bb9, label %bb7

bb7:                                              ; preds = %bb5
  store i32 %tmp3, i32 addrspace(1)* %out
  br label %bb9

bb9:                                              ; preds = %bb8, %bb4
  ret void
}

; GCN-LABEL: {{^}}uniform_if_scc_i64_eq:
; VI-DAG: s_cmp_eq_u64 s{{\[[0-9]+:[0-9]+\]}}, 0
; GCN-DAG: s_mov_b32 [[S_VAL:s[0-9]+]], 0
; SI-DAG: v_cmp_eq_u64_e64
; SI: s_cbranch_vccnz [[IF_LABEL:.L[0-9_A-Za-z]+]]

; VI: s_cbranch_scc1 [[IF_LABEL:.L[0-9_A-Za-z]+]]

; Fall-through to the else
; GCN: s_mov_b32 [[S_VAL]], 1

; GCN: [[IF_LABEL]]:
; GCN: v_mov_b32_e32 [[V_VAL:v[0-9]+]], [[S_VAL]]
; GCN: buffer_store_dword [[V_VAL]]
define amdgpu_kernel void @uniform_if_scc_i64_eq(i64 %cond, i32 addrspace(1)* %out) {
entry:
  %cmp0 = icmp eq i64 %cond, 0
  br i1 %cmp0, label %if, label %else

if:
  br label %done

else:
  br label %done

done:
  %value = phi i32 [0, %if], [1, %else]
  store i32 %value, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}uniform_if_scc_i64_ne:
; VI-DAG: s_cmp_lg_u64 s{{\[[0-9]+:[0-9]+\]}}, 0
; GCN-DAG: s_mov_b32 [[S_VAL:s[0-9]+]], 0

; SI-DAG: v_cmp_ne_u64_e64
; SI: s_cbranch_vccnz [[IF_LABEL:.L[0-9_A-Za-z]+]]

; VI: s_cbranch_scc1 [[IF_LABEL:.L[0-9_A-Za-z]+]]

; Fall-through to the else
; GCN: s_mov_b32 [[S_VAL]], 1

; GCN: [[IF_LABEL]]:
; GCN: v_mov_b32_e32 [[V_VAL:v[0-9]+]], [[S_VAL]]
; GCN: buffer_store_dword [[V_VAL]]
define amdgpu_kernel void @uniform_if_scc_i64_ne(i64 %cond, i32 addrspace(1)* %out) {
entry:
  %cmp0 = icmp ne i64 %cond, 0
  br i1 %cmp0, label %if, label %else

if:
  br label %done

else:
  br label %done

done:
  %value = phi i32 [0, %if], [1, %else]
  store i32 %value, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}uniform_if_scc_i64_sgt:
; GCN-DAG: s_mov_b32 [[S_VAL:s[0-9]+]], 0
; GCN-DAG: v_cmp_gt_i64_e64
; GCN: s_cbranch_vccnz [[IF_LABEL:.L[0-9_A-Za-z]+]]

; Fall-through to the else
; GCN: s_mov_b32 [[S_VAL]], 1

; GCN: [[IF_LABEL]]:
; GCN: v_mov_b32_e32 [[V_VAL:v[0-9]+]], [[S_VAL]]
; GCN: buffer_store_dword [[V_VAL]]
define amdgpu_kernel void @uniform_if_scc_i64_sgt(i64 %cond, i32 addrspace(1)* %out) {
entry:
  %cmp0 = icmp sgt i64 %cond, 0
  br i1 %cmp0, label %if, label %else

if:
  br label %done

else:
  br label %done

done:
  %value = phi i32 [0, %if], [1, %else]
  store i32 %value, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}move_to_valu_i64_eq:
; GCN: v_cmp_eq_u64_e32
define amdgpu_kernel void @move_to_valu_i64_eq(i32 addrspace(1)* %out) {
  %cond = load volatile i64, i64 addrspace(3)* undef
  %cmp0 = icmp eq i64 %cond, 0
  br i1 %cmp0, label %if, label %else

if:
  br label %done

else:
  br label %done

done:
  %value = phi i32 [0, %if], [1, %else]
  store i32 %value, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}move_to_valu_i64_ne:
; GCN: v_cmp_ne_u64_e32
define amdgpu_kernel void @move_to_valu_i64_ne(i32 addrspace(1)* %out) {
  %cond = load volatile i64, i64 addrspace(3)* undef
  %cmp0 = icmp ne i64 %cond, 0
  br i1 %cmp0, label %if, label %else

if:
  br label %done

else:
  br label %done

done:
  %value = phi i32 [0, %if], [1, %else]
  store i32 %value, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}move_to_valu_vgpr_operand_phi:
; GCN: v_add_{{[iu]}}32_e32
; GCN: ds_write_b32
define void @move_to_valu_vgpr_operand_phi(i32 addrspace(3)* %out) {
bb0:
  br label %bb1

bb1:                                              ; preds = %bb3, %bb0
  %tmp0 = phi i32 [ 8, %bb0 ], [ %tmp4, %bb3 ]
  %tmp1 = add nsw i32 %tmp0, -1
  %tmp2 = getelementptr inbounds i32, i32 addrspace(3)* %out, i32 %tmp1
  br i1 undef, label %bb2, label %bb3

bb2:                                              ; preds = %bb1
  store volatile i32 1, i32 addrspace(3)* %tmp2, align 4
  br label %bb3

bb3:                                              ; preds = %bb2, %bb1
  %tmp4 = add nsw i32 %tmp0, 2
  br label %bb1
}

declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone }
