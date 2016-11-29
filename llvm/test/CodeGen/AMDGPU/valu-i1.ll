; RUN: llc -march=amdgcn -verify-machineinstrs -enable-misched -asm-verbose < %s | FileCheck -check-prefix=SI %s

declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone

; SI-LABEL: {{^}}test_if:
; Make sure the i1 values created by the cfg structurizer pass are
; moved using VALU instructions


; waitcnt should be inserted after exec modification
; SI: v_cmp_lt_i32_e32 vcc, 0,
; SI-NEXT: s_and_saveexec_b64 [[SAVE1:s\[[0-9]+:[0-9]+\]]], vcc
; SI-NEXT: s_xor_b64 [[SAVE2:s\[[0-9]+:[0-9]+\]]], exec, [[SAVE1]]
; SI-NEXT: s_waitcnt lgkmcnt(0)
; SI-NEXT: ; mask branch [[FLOW_BB:BB[0-9]+_[0-9]+]]
; SI-NEXT: s_cbranch_execz [[FLOW_BB]]

; SI-NEXT: BB{{[0-9]+}}_1: ; %LeafBlock3
; SI-NOT: s_mov_b64 s[{{[0-9]:[0-9]}}], -1
; SI: v_mov_b32_e32 v{{[0-9]}}, -1
; SI: s_and_saveexec_b64
; SI-NEXT: s_xor_b64
; SI-NEXT: ; mask branch

; v_mov should be after exec modification
; SI: [[FLOW_BB]]:
; SI-NEXT: s_or_saveexec_b64 [[SAVE3:s\[[0-9]+:[0-9]+\]]], [[SAVE2]]
; SI-NEXT: v_mov_b32_e32 v{{[0-9]+}}
; SI-NEXT: s_xor_b64 exec, exec, [[SAVE3]]
; SI-NEXT: ; mask branch
;
define void @test_if(i32 %b, i32 addrspace(1)* %src, i32 addrspace(1)* %dst) #1 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  switch i32 %tid, label %default [
    i32 0, label %case0
    i32 1, label %case1
  ]

case0:
  %arrayidx1 = getelementptr i32, i32 addrspace(1)* %dst, i32 %b
  store i32 13, i32 addrspace(1)* %arrayidx1, align 4
  br label %end

case1:
  %arrayidx5 = getelementptr i32, i32 addrspace(1)* %dst, i32 %b
  store i32 17, i32 addrspace(1)* %arrayidx5, align 4
  br label %end

default:
  %cmp8 = icmp eq i32 %tid, 2
  %arrayidx10 = getelementptr i32, i32 addrspace(1)* %dst, i32 %b
  br i1 %cmp8, label %if, label %else

if:
  store i32 19, i32 addrspace(1)* %arrayidx10, align 4
  br label %end

else:
  store i32 21, i32 addrspace(1)* %arrayidx10, align 4
  br label %end

end:
  ret void
}

; SI-LABEL: @simple_test_v_if
; SI: v_cmp_ne_u32_e32 vcc, 0, v{{[0-9]+}}
; SI: s_and_saveexec_b64 [[BR_SREG:s\[[0-9]+:[0-9]+\]]], vcc
; SI: s_xor_b64 [[BR_SREG]], exec, [[BR_SREG]]

; SI: BB{{[0-9]+_[0-9]+}}:
; SI: buffer_store_dword
; SI: s_endpgm

; SI: BB1_2:
; SI: s_or_b64 exec, exec, [[BR_SREG]]
; SI: s_endpgm
define void @simple_test_v_if(i32 addrspace(1)* %dst, i32 addrspace(1)* %src) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %is.0 = icmp ne i32 %tid, 0
  br i1 %is.0, label %store, label %exit

store:
  %gep = getelementptr i32, i32 addrspace(1)* %dst, i32 %tid
  store i32 999, i32 addrspace(1)* %gep
  ret void

exit:
  ret void
}

; SI-LABEL: {{^}}simple_test_v_loop:
; SI: v_cmp_ne_u32_e32 vcc, 0, v{{[0-9]+}}
; SI: s_and_saveexec_b64 [[BR_SREG:s\[[0-9]+:[0-9]+\]]], vcc
; SI: s_xor_b64 [[BR_SREG]], exec, [[BR_SREG]]
; SI: s_cbranch_execz [[LABEL_EXIT:BB[0-9]+_[0-9]+]]

; SI: s_mov_b64 {{s\[[0-9]+:[0-9]+\]}}, 0{{$}}

; SI: [[LABEL_LOOP:BB[0-9]+_[0-9]+]]:
; SI: buffer_load_dword
; SI-DAG: buffer_store_dword
; SI-DAG: s_cmpk_eq_i32 s{{[0-9]+}}, 0x100
; SI: s_cbranch_scc0 [[LABEL_LOOP]]
; SI: [[LABEL_EXIT]]:
; SI: s_endpgm

define void @simple_test_v_loop(i32 addrspace(1)* %dst, i32 addrspace(1)* %src) #1 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %is.0 = icmp ne i32 %tid, 0
  %limit = add i32 %tid, 64
  br i1 %is.0, label %loop, label %exit

loop:
  %i = phi i32 [%tid, %entry], [%i.inc, %loop]
  %gep.src = getelementptr i32, i32 addrspace(1)* %src, i32 %i
  %gep.dst = getelementptr i32, i32 addrspace(1)* %dst, i32 %i
  %load = load i32, i32 addrspace(1)* %src
  store i32 %load, i32 addrspace(1)* %gep.dst
  %i.inc = add nsw i32 %i, 1
  %cmp = icmp eq i32 %limit, %i.inc
  br i1 %cmp, label %exit, label %loop

exit:
  ret void
}

; SI-LABEL: {{^}}multi_vcond_loop:

; Load loop limit from buffer
; Branch to exit if uniformly not taken
; SI: ; BB#0:
; SI: buffer_load_dword [[VBOUND:v[0-9]+]]
; SI: v_cmp_lt_i32_e32 vcc
; SI: s_and_saveexec_b64 [[OUTER_CMP_SREG:s\[[0-9]+:[0-9]+\]]], vcc
; SI: s_xor_b64 [[OUTER_CMP_SREG]], exec, [[OUTER_CMP_SREG]]
; SI: s_cbranch_execz [[LABEL_EXIT:BB[0-9]+_[0-9]+]]

; Initialize inner condition to false
; SI: BB{{[0-9]+_[0-9]+}}: ; %bb10.preheader
; SI: s_mov_b64 [[ZERO:s\[[0-9]+:[0-9]+\]]], 0{{$}}
; SI: s_mov_b64 [[COND_STATE:s\[[0-9]+:[0-9]+\]]], [[ZERO]]

; Clear exec bits for workitems that load -1s
; SI: [[LABEL_LOOP:BB[0-9]+_[0-9]+]]:
; SI: buffer_load_dword [[B:v[0-9]+]]
; SI: buffer_load_dword [[A:v[0-9]+]]
; SI-DAG: v_cmp_ne_u32_e64 [[NEG1_CHECK_0:s\[[0-9]+:[0-9]+\]]], -1, [[A]]
; SI-DAG: v_cmp_ne_u32_e32 [[NEG1_CHECK_1:vcc]], -1, [[B]]
; SI: s_and_b64 [[ORNEG1:s\[[0-9]+:[0-9]+\]]], [[NEG1_CHECK_1]], [[NEG1_CHECK_0]]
; SI: s_and_saveexec_b64 [[ORNEG2:s\[[0-9]+:[0-9]+\]]], [[ORNEG1]]
; SI: s_xor_b64 [[ORNEG3:s\[[0-9]+:[0-9]+\]]], exec, [[ORNEG2]]
; SI: s_cbranch_execz [[LABEL_FLOW:BB[0-9]+_[0-9]+]]

; SI: BB{{[0-9]+_[0-9]+}}: ; %bb20
; SI: buffer_store_dword
; SI: v_cmp_ge_i64_e32 [[CMP:s\[[0-9]+:[0-9]+\]|vcc]]
; SI: s_or_b64 [[TMP:s\[[0-9]+:[0-9]+\]]], [[CMP]], [[COND_STATE]]

; SI: [[LABEL_FLOW]]:
; SI-NEXT: ; in Loop: Header=[[LABEL_LOOP]]
; SI-NEXT: s_or_b64 exec, exec, [[ORNEG3]]
; SI-NEXT: s_or_b64 [[COND_STATE]], [[ORNEG3]], [[TMP]]
; SI-NEXT: s_andn2_b64 exec, exec, [[COND_STATE]]
; SI-NEXT: s_cbranch_execnz [[LABEL_LOOP]]

; SI: BB#5
; SI: s_or_b64 exec, exec, [[COND_STATE]]

; SI: [[LABEL_EXIT]]:
; SI-NOT: [[COND_STATE]]
; SI: s_endpgm

define void @multi_vcond_loop(i32 addrspace(1)* noalias nocapture %arg, i32 addrspace(1)* noalias nocapture readonly %arg1, i32 addrspace(1)* noalias nocapture readonly %arg2, i32 addrspace(1)* noalias nocapture readonly %arg3) #1 {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %tmp4 = sext i32 %tmp to i64
  %tmp5 = getelementptr inbounds i32, i32 addrspace(1)* %arg3, i64 %tmp4
  %tmp6 = load i32, i32 addrspace(1)* %tmp5, align 4
  %tmp7 = icmp sgt i32 %tmp6, 0
  %tmp8 = sext i32 %tmp6 to i64
  br i1 %tmp7, label %bb10, label %bb26

bb10:                                             ; preds = %bb, %bb20
  %tmp11 = phi i64 [ %tmp23, %bb20 ], [ 0, %bb ]
  %tmp12 = add nsw i64 %tmp11, %tmp4
  %tmp13 = getelementptr inbounds i32, i32 addrspace(1)* %arg1, i64 %tmp12
  %tmp14 = load i32, i32 addrspace(1)* %tmp13, align 4
  %tmp15 = getelementptr inbounds i32, i32 addrspace(1)* %arg2, i64 %tmp12
  %tmp16 = load i32, i32 addrspace(1)* %tmp15, align 4
  %tmp17 = icmp ne i32 %tmp14, -1
  %tmp18 = icmp ne i32 %tmp16, -1
  %tmp19 = and i1 %tmp17, %tmp18
  br i1 %tmp19, label %bb20, label %bb26

bb20:                                             ; preds = %bb10
  %tmp21 = add nsw i32 %tmp16, %tmp14
  %tmp22 = getelementptr inbounds i32, i32 addrspace(1)* %arg, i64 %tmp12
  store i32 %tmp21, i32 addrspace(1)* %tmp22, align 4
  %tmp23 = add nuw nsw i64 %tmp11, 1
  %tmp24 = icmp slt i64 %tmp23, %tmp8
  br i1 %tmp24, label %bb10, label %bb26

bb26:                                             ; preds = %bb10, %bb20, %bb
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
