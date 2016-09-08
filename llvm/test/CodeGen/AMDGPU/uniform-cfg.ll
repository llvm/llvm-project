; RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck --check-prefix=SI %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck --check-prefix=SI %s

; SI-LABEL: {{^}}uniform_if_scc:
; SI-DAG: s_cmp_eq_i32 s{{[0-9]+}}, 0
; SI-DAG: v_mov_b32_e32 [[STORE_VAL:v[0-9]+]], 0
; SI: s_cbranch_scc1 [[IF_LABEL:[0-9_A-Za-z]+]]

; Fall-through to the else
; SI: v_mov_b32_e32 [[STORE_VAL]], 1

; SI: [[IF_LABEL]]:
; SI: buffer_store_dword [[STORE_VAL]]
define void @uniform_if_scc(i32 %cond, i32 addrspace(1)* %out) {
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

; SI-LABEL: {{^}}uniform_if_vcc:
; FIXME: We could use _e32 here if we re-used the 0 from [[STORE_VAL]], and
; also scheduled the write first.
; SI-DAG: v_cmp_eq_f32_e64 [[COND:vcc|s\[[0-9]+:[0-9]+\]]], s{{[0-9]+}}, 0{{$}}
; SI-DAG: s_and_b64 vcc, exec, [[COND]]
; SI-DAG: v_mov_b32_e32 [[STORE_VAL:v[0-9]+]], 0
; SI: s_cbranch_vccnz [[IF_LABEL:[0-9_A-Za-z]+]]

; Fall-through to the else
; SI: v_mov_b32_e32 [[STORE_VAL]], 1

; SI: [[IF_LABEL]]:
; SI: buffer_store_dword [[STORE_VAL]]
define void @uniform_if_vcc(float %cond, i32 addrspace(1)* %out) {
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

; SI-LABEL: {{^}}uniform_if_swap_br_targets_scc:
; SI-DAG: s_cmp_lg_i32 s{{[0-9]+}}, 0
; SI-DAG: v_mov_b32_e32 [[STORE_VAL:v[0-9]+]], 0
; SI: s_cbranch_scc1 [[IF_LABEL:[0-9_A-Za-z]+]]

; Fall-through to the else
; SI: v_mov_b32_e32 [[STORE_VAL]], 1

; SI: [[IF_LABEL]]:
; SI: buffer_store_dword [[STORE_VAL]]
define void @uniform_if_swap_br_targets_scc(i32 %cond, i32 addrspace(1)* %out) {
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

; SI-LABEL: {{^}}uniform_if_swap_br_targets_vcc:
; FIXME: We could use _e32 here if we re-used the 0 from [[STORE_VAL]], and
; also scheduled the write first.
; SI-DAG: v_cmp_neq_f32_e64 [[COND:vcc|s\[[0-9]+:[0-9]+\]]], s{{[0-9]+}}, 0{{$}}
; SI-DAG: s_and_b64 vcc, exec, [[COND]]
; SI-DAG: v_mov_b32_e32 [[STORE_VAL:v[0-9]+]], 0
; SI: s_cbranch_vccnz [[IF_LABEL:[0-9_A-Za-z]+]]

; Fall-through to the else
; SI: v_mov_b32_e32 [[STORE_VAL]], 1

; SI: [[IF_LABEL]]:
; SI: buffer_store_dword [[STORE_VAL]]
define void @uniform_if_swap_br_targets_vcc(float %cond, i32 addrspace(1)* %out) {
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

; SI-LABEL: {{^}}uniform_if_move_valu:
; SI: v_add_f32_e32 [[CMP:v[0-9]+]]
; Using a floating-point value in an integer compare will cause the compare to
; be selected for the SALU and then later moved to the VALU.
; SI: v_cmp_ne_i32_e32 [[COND:vcc|s\[[0-9]+:[0-9]+\]]], 5, [[CMP]]
; SI: s_and_b64 vcc, exec, [[COND]]
; SI: s_cbranch_vccnz [[ENDIF_LABEL:[0-9_A-Za-z]+]]
; SI: buffer_store_dword
; SI: [[ENDIF_LABEL]]:
; SI: s_endpgm
define void @uniform_if_move_valu(i32 addrspace(1)* %out, float %a) {
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

; SI-LABEL: {{^}}uniform_if_move_valu_commute:
; SI: v_add_f32_e32 [[CMP:v[0-9]+]]
; Using a floating-point value in an integer compare will cause the compare to
; be selected for the SALU and then later moved to the VALU.
; SI: v_cmp_gt_u32_e32 [[COND:vcc|s\[[0-9]+:[0-9]+\]]], 6, [[CMP]]
; SI: s_and_b64 vcc, exec, [[COND]]
; SI: s_cbranch_vccnz [[ENDIF_LABEL:[0-9_A-Za-z]+]]
; SI: buffer_store_dword
; SI: [[ENDIF_LABEL]]:
; SI: s_endpgm
define void @uniform_if_move_valu_commute(i32 addrspace(1)* %out, float %a) {
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


; SI-LABEL: {{^}}uniform_if_else_ret:
; SI: s_cmp_lg_i32 s{{[0-9]+}}, 0
; SI-NEXT: s_cbranch_scc0 [[IF_LABEL:[0-9_A-Za-z]+]]

; SI: v_mov_b32_e32 [[TWO:v[0-9]+]], 2
; SI: buffer_store_dword [[TWO]]
; SI: s_endpgm

; SI: {{^}}[[IF_LABEL]]:
; SI: v_mov_b32_e32 [[ONE:v[0-9]+]], 1
; SI: buffer_store_dword [[ONE]]
; SI: s_endpgm
define void @uniform_if_else_ret(i32 addrspace(1)* nocapture %out, i32 %a) {
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

; SI-LABEL: {{^}}uniform_if_else:
; SI: s_cmp_lg_i32 s{{[0-9]+}}, 0
; SI-NEXT: s_cbranch_scc0 [[IF_LABEL:[0-9_A-Za-z]+]]

; SI: v_mov_b32_e32 [[TWO:v[0-9]+]], 2
; SI: buffer_store_dword [[TWO]]
; SI: s_branch [[ENDIF_LABEL:[0-9_A-Za-z]+]]

; SI: [[IF_LABEL]]:
; SI: v_mov_b32_e32 [[ONE:v[0-9]+]], 1
; SI: buffer_store_dword [[ONE]]

; SI: [[ENDIF_LABEL]]:
; SI: v_mov_b32_e32 [[THREE:v[0-9]+]], 3
; SI: buffer_store_dword [[THREE]]
; SI: s_endpgm
define void @uniform_if_else(i32 addrspace(1)* nocapture %out0, i32 addrspace(1)* nocapture %out1, i32 %a) {
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

; SI-LABEL: {{^}}icmp_2_users:
; SI: s_cmp_lt_i32 s{{[0-9]+}}, 1
; SI: s_cbranch_scc1 [[LABEL:[a-zA-Z0-9_]+]]
; SI: buffer_store_dword
; SI: [[LABEL]]:
; SI: s_endpgm
define void @icmp_2_users(i32 addrspace(1)* %out, i32 %cond) {
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

; SI-LABEL: {{^}}icmp_users_different_blocks:
; SI: s_load_dword [[COND:s[0-9]+]]
; SI: s_cmp_lt_i32 [[COND]], 1
; SI: s_cbranch_scc1 [[EXIT:[A-Za-z0-9_]+]]
; SI: v_cmp_gt_i32_e64 [[MASK:s\[[0-9]+:[0-9]+\]]], [[COND]], 0{{$}}
; SI: s_and_b64 vcc, exec, [[MASK]]
; SI: s_cbranch_vccnz [[EXIT]]
; SI: buffer_store
; SI: {{^}}[[EXIT]]:
; SI: s_endpgm
define void @icmp_users_different_blocks(i32 %cond0, i32 %cond1, i32 addrspace(1)* %out) {
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
; SI: {{^}}[[LOOP_LABEL:[A-Z0-9_a-z]+]]:
; FIXME: We need to teach SIFixSGPRCopies about uniform branches so we
;        get s_add_i32 here.
; SI: v_add_i32_e32 [[I:v[0-9]+]], vcc, -1, v{{[0-9]+}}
; SI: v_cmp_ne_i32_e32 vcc, 0, [[I]]
; SI: s_and_b64 vcc, exec, vcc
; SI: s_cbranch_vccnz [[LOOP_LABEL]]
; SI: s_endpgm
define void @uniform_loop(i32 addrspace(1)* %out, i32 %a) {
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

; SI-LABEL: {{^}}uniform_inside_divergent:
; SI: v_cmp_gt_u32_e32 vcc, 16, v{{[0-9]+}}
; SI: s_and_saveexec_b64 [[MASK:s\[[0-9]+:[0-9]+\]]], vcc
; SI: s_xor_b64  [[MASK1:s\[[0-9]+:[0-9]+\]]], exec, [[MASK]]
; SI: s_cbranch_execz [[ENDIF_LABEL:[0-9_A-Za-z]+]]
; SI: s_cmp_lg_i32 {{s[0-9]+}}, 0
; SI: s_cbranch_scc1 [[ENDIF_LABEL]]
; SI: v_mov_b32_e32 [[ONE:v[0-9]+]], 1
; SI: buffer_store_dword [[ONE]]
define void @uniform_inside_divergent(i32 addrspace(1)* %out, i32 %cond) {
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

; SI-LABEL: {{^}}divergent_inside_uniform:
; SI: s_cmp_lg_i32 s{{[0-9]+}}, 0
; SI: s_cbranch_scc1 [[ENDIF_LABEL:[0-9_A-Za-z]+]]
; SI: v_cmp_gt_u32_e32 vcc, 16, v{{[0-9]+}}
; SI: s_and_saveexec_b64 [[MASK:s\[[0-9]+:[0-9]+\]]], vcc
; SI: s_xor_b64  [[MASK1:s\[[0-9]+:[0-9]+\]]], exec, [[MASK]]
; SI: v_mov_b32_e32 [[ONE:v[0-9]+]], 1
; SI: buffer_store_dword [[ONE]]
; SI: [[ENDIF_LABEL]]:
; SI: s_endpgm
define void @divergent_inside_uniform(i32 addrspace(1)* %out, i32 %cond) {
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

; SI-LABEL: {{^}}divergent_if_uniform_if:
; SI: v_cmp_eq_i32_e32 vcc, 0, v0
; SI: s_and_saveexec_b64 [[MASK:s\[[0-9]+:[0-9]+\]]], vcc
; SI: s_xor_b64 [[MASK:s\[[0-9]+:[0-9]+\]]], exec, [[MASK]]
; SI: v_mov_b32_e32 [[ONE:v[0-9]+]], 1
; SI: buffer_store_dword [[ONE]]
; SI: s_or_b64 exec, exec, [[MASK]]
; SI: s_cmp_lg_i32 s{{[0-9]+}}, 0
; SI: s_cbranch_scc1 [[EXIT:[A-Z0-9_]+]]
; SI: v_mov_b32_e32 [[TWO:v[0-9]+]], 2
; SI: buffer_store_dword [[TWO]]
; SI: [[EXIT]]:
; SI: s_endpgm
define void @divergent_if_uniform_if(i32 addrspace(1)* %out, i32 %cond) {
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

; SI-LABEL: {{^}}cse_uniform_condition_different_blocks:
; SI: s_load_dword [[COND:s[0-9]+]]
; SI: s_cmp_lt_i32 [[COND]], 1
; SI: s_cbranch_scc1 BB[[FNNUM:[0-9]+]]_3

; SI: BB#1:
; SI-NOT: cmp
; SI: buffer_load_dword
; SI: buffer_store_dword
; SI: s_cbranch_scc1 BB[[FNNUM]]_3

; SI: BB[[FNNUM]]_3:
; SI: s_endpgm
define void @cse_uniform_condition_different_blocks(i32 %cond, i32 addrspace(1)* %out) {
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

declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { readnone }
