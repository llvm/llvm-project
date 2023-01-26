; RUN: llc -march=amdgcn -verify-machineinstrs -enable-misched -asm-verbose -disable-block-placement -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck -check-prefix=SI %s

declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone

; SI-LABEL: {{^}}test_if:
; Make sure the i1 values created by the cfg structurizer pass are
; moved using VALU instructions


; waitcnt should be inserted after exec modification
; SI:      v_cmp_lt_i32_e32 vcc, 1,
; SI-NEXT: s_mov_b64 {{s\[[0-9]+:[0-9]+\]}}, 0
; SI-NEXT: s_mov_b64 {{s\[[0-9]+:[0-9]+\]}}, 0
; SI-NEXT: s_and_saveexec_b64 [[SAVE1:s\[[0-9]+:[0-9]+\]]], vcc
; SI-NEXT: s_xor_b64 [[SAVE2:s\[[0-9]+:[0-9]+\]]], exec, [[SAVE1]]
; SI-NEXT: s_cbranch_execz [[FLOW_BB:.LBB[0-9]+_[0-9]+]]

; SI-NEXT: ; %bb.{{[0-9]+}}: ; %LeafBlock3
; SI:      s_mov_b64 s[{{[0-9]:[0-9]}}], -1
; SI:      s_and_saveexec_b64
; SI-NEXT: s_cbranch_execnz

; v_mov should be after exec modification
; SI: [[FLOW_BB]]:
; SI-NEXT: s_andn2_saveexec_b64 [[SAVE2]], [[SAVE2]]
;
define amdgpu_kernel void @test_if(i32 %b, ptr addrspace(1) %src, ptr addrspace(1) %dst) #1 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  switch i32 %tid, label %default [
    i32 1, label %case1
    i32 2, label %case2
  ]

case1:
  %arrayidx1 = getelementptr i32, ptr addrspace(1) %dst, i32 %b
  store i32 13, ptr addrspace(1) %arrayidx1, align 4
  br label %end

case2:
  %arrayidx5 = getelementptr i32, ptr addrspace(1) %dst, i32 %b
  store i32 17, ptr addrspace(1) %arrayidx5, align 4
  br label %end

default:
  %cmp8 = icmp eq i32 %tid, 2
  %arrayidx10 = getelementptr i32, ptr addrspace(1) %dst, i32 %b
  br i1 %cmp8, label %if, label %else

if:
  store i32 19, ptr addrspace(1) %arrayidx10, align 4
  br label %end

else:
  store i32 21, ptr addrspace(1) %arrayidx10, align 4
  br label %end

end:
  ret void
}

; SI-LABEL: {{^}}simple_test_v_if:
; SI: v_cmp_ne_u32_e32 vcc, 0, v{{[0-9]+}}
; SI: s_and_saveexec_b64 [[BR_SREG:s\[[0-9]+:[0-9]+\]]], vcc
; SI-NEXT: s_cbranch_execz [[EXIT:.LBB[0-9]+_[0-9]+]]

; SI-NEXT: ; %bb.{{[0-9]+}}:
; SI: buffer_store_dword

; SI-NEXT: {{^}}[[EXIT]]:
; SI: s_endpgm
define amdgpu_kernel void @simple_test_v_if(ptr addrspace(1) %dst, ptr addrspace(1) %src) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %is.0 = icmp ne i32 %tid, 0
  br i1 %is.0, label %then, label %exit

then:
  %gep = getelementptr i32, ptr addrspace(1) %dst, i32 %tid
  store i32 999, ptr addrspace(1) %gep
  br label %exit

exit:
  ret void
}

; FIXME: It would be better to endpgm in the then block.

; SI-LABEL: {{^}}simple_test_v_if_ret_else_ret:
; SI: v_cmp_ne_u32_e32 vcc, 0, v{{[0-9]+}}
; SI: s_and_saveexec_b64 [[BR_SREG:s\[[0-9]+:[0-9]+\]]], vcc
; SI-NEXT: s_cbranch_execz [[EXIT:.LBB[0-9]+_[0-9]+]]

; SI-NEXT: ; %bb.{{[0-9]+}}:
; SI: buffer_store_dword

; SI-NEXT: {{^}}[[EXIT]]:
; SI: s_endpgm
define amdgpu_kernel void @simple_test_v_if_ret_else_ret(ptr addrspace(1) %dst, ptr addrspace(1) %src) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %is.0 = icmp ne i32 %tid, 0
  br i1 %is.0, label %then, label %exit

then:
  %gep = getelementptr i32, ptr addrspace(1) %dst, i32 %tid
  store i32 999, ptr addrspace(1) %gep
  ret void

exit:
  ret void
}

; Final block has more than a ret to execute. This was miscompiled
; before function exit blocks were unified since the endpgm would
; terminate the then wavefront before reaching the store.

; SI-LABEL: {{^}}simple_test_v_if_ret_else_code_ret:
; SI: v_cmp_eq_u32_e32 vcc, 0, v{{[0-9]+}}
; SI: s_and_saveexec_b64 [[BR_SREG:s\[[0-9]+:[0-9]+\]]], vcc
; SI: s_xor_b64 [[BR_SREG]], exec, [[BR_SREG]]
; SI: s_cbranch_execnz [[EXIT:.LBB[0-9]+_[0-9]+]]

; SI-NEXT: {{^.LBB[0-9]+_[0-9]+}}: ; %Flow
; SI-NEXT: s_andn2_saveexec_b64 [[BR_SREG]], [[BR_SREG]]
; SI-NEXT: s_cbranch_execz [[UNIFIED_RETURN:.LBB[0-9]+_[0-9]+]]

; SI-NEXT: ; %bb.{{[0-9]+}}: ; %then
; SI: s_waitcnt
; SI-NEXT: buffer_store_dword

; SI-NEXT: {{^}}[[UNIFIED_RETURN]]: ; %UnifiedReturnBlock
; SI: s_endpgm

; SI-NEXT: {{^}}[[EXIT]]:
; SI: ds_write_b32
define amdgpu_kernel void @simple_test_v_if_ret_else_code_ret(ptr addrspace(1) %dst, ptr addrspace(1) %src) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %is.0 = icmp ne i32 %tid, 0
  br i1 %is.0, label %then, label %exit

then:
  %gep = getelementptr i32, ptr addrspace(1) %dst, i32 %tid
  store i32 999, ptr addrspace(1) %gep
  ret void

exit:
  store volatile i32 7, ptr addrspace(3) undef
  ret void
}

; SI-LABEL: {{^}}simple_test_v_loop:
; SI: v_cmp_ne_u32_e32 vcc, 0, v{{[0-9]+}}
; SI: s_and_saveexec_b64 [[BR_SREG:s\[[0-9]+:[0-9]+\]]], vcc
; SI-NEXT: s_cbranch_execz [[LABEL_EXIT:.LBB[0-9]+_[0-9]+]]

; SI: s_mov_b64 {{s\[[0-9]+:[0-9]+\]}}, 0{{$}}

; SI: [[LABEL_LOOP:.LBB[0-9]+_[0-9]+]]:
; SI: buffer_load_dword
; SI-DAG: buffer_store_dword
; SI-DAG: s_cmpk_lg_i32 s{{[0-9]+}}, 0x100
; SI: s_cbranch_scc1 [[LABEL_LOOP]]
; SI: [[LABEL_EXIT]]:
; SI: s_endpgm
define amdgpu_kernel void @simple_test_v_loop(ptr addrspace(1) %dst, ptr addrspace(1) %src) #1 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %is.0 = icmp ne i32 %tid, 0
  %limit = add i32 %tid, 64
  br i1 %is.0, label %loop, label %exit

loop:
  %i = phi i32 [%tid, %entry], [%i.inc, %loop]
  %gep.src = getelementptr i32, ptr addrspace(1) %src, i32 %i
  %gep.dst = getelementptr i32, ptr addrspace(1) %dst, i32 %i
  %load = load i32, ptr addrspace(1) %src
  store i32 %load, ptr addrspace(1) %gep.dst
  %i.inc = add nsw i32 %i, 1
  %cmp = icmp eq i32 %limit, %i.inc
  br i1 %cmp, label %exit, label %loop

exit:
  ret void
}

; SI-LABEL: {{^}}multi_vcond_loop:

; Load loop limit from buffer
; Branch to exit if uniformly not taken
; SI: ; %bb.0:
; SI: buffer_load_dword [[VBOUND:v[0-9]+]]
; SI: v_cmp_lt_i32_e32 vcc
; SI: s_and_saveexec_b64 [[OUTER_CMP_SREG:s\[[0-9]+:[0-9]+\]]], vcc
; SI-NEXT: s_cbranch_execz [[LABEL_EXIT:.LBB[0-9]+_[0-9]+]]

; Initialize inner condition to false
; SI: ; %bb.{{[0-9]+}}: ; %bb10.preheader
; SI: s_mov_b64 [[COND_STATE:s\[[0-9]+:[0-9]+\]]], 0{{$}}

; Clear exec bits for workitems that load -1s
; SI: .L[[LABEL_LOOP:BB[0-9]+_[0-9]+]]:
; SI: buffer_load_dword [[B:v[0-9]+]]
; SI: buffer_load_dword [[A:v[0-9]+]]
; SI-DAG: v_cmp_ne_u32_e64 [[NEG1_CHECK_0:s\[[0-9]+:[0-9]+\]]], -1, [[A]]
; SI-DAG: v_cmp_ne_u32_e32 [[NEG1_CHECK_1:vcc]], -1, [[B]]
; SI: s_and_b64 [[ORNEG1:s\[[0-9]+:[0-9]+\]]], [[NEG1_CHECK_1]], [[NEG1_CHECK_0]]
; SI: s_and_saveexec_b64 [[ORNEG2:s\[[0-9]+:[0-9]+\]]], [[ORNEG1]]
; SI: s_cbranch_execz [[LABEL_FLOW:.LBB[0-9]+_[0-9]+]]

; SI: ; %bb.{{[0-9]+}}: ; %bb20
; SI: buffer_store_dword

; SI: [[LABEL_FLOW]]:
; SI-NEXT: ; in Loop: Header=[[LABEL_LOOP]]
; SI-NEXT: s_or_b64 exec, exec, [[ORNEG2]]
; SI-NEXT: s_and_b64 [[TMP1:s\[[0-9]+:[0-9]+\]]],
; SI-NEXT: s_or_b64 [[COND_STATE]], [[TMP1]], [[COND_STATE]]
; SI-NEXT: s_andn2_b64 exec, exec, [[COND_STATE]]
; SI-NEXT: s_cbranch_execnz .L[[LABEL_LOOP]]

; SI: [[LABEL_EXIT]]:
; SI-NOT: [[COND_STATE]]
; SI: s_endpgm
define amdgpu_kernel void @multi_vcond_loop(ptr addrspace(1) noalias nocapture %arg, ptr addrspace(1) noalias nocapture readonly %arg1, ptr addrspace(1) noalias nocapture readonly %arg2, ptr addrspace(1) noalias nocapture readonly %arg3) #1 {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %tmp4 = sext i32 %tmp to i64
  %tmp5 = getelementptr inbounds i32, ptr addrspace(1) %arg3, i64 %tmp4
  %tmp6 = load i32, ptr addrspace(1) %tmp5, align 4
  %tmp7 = icmp sgt i32 %tmp6, 0
  %tmp8 = sext i32 %tmp6 to i64
  br i1 %tmp7, label %bb10, label %bb26

bb10:                                             ; preds = %bb, %bb20
  %tmp11 = phi i64 [ %tmp23, %bb20 ], [ 0, %bb ]
  %tmp12 = add nsw i64 %tmp11, %tmp4
  %tmp13 = getelementptr inbounds i32, ptr addrspace(1) %arg1, i64 %tmp12
  %tmp14 = load i32, ptr addrspace(1) %tmp13, align 4
  %tmp15 = getelementptr inbounds i32, ptr addrspace(1) %arg2, i64 %tmp12
  %tmp16 = load i32, ptr addrspace(1) %tmp15, align 4
  %tmp17 = icmp ne i32 %tmp14, -1
  %tmp18 = icmp ne i32 %tmp16, -1
  %tmp19 = and i1 %tmp17, %tmp18
  br i1 %tmp19, label %bb20, label %bb26

bb20:                                             ; preds = %bb10
  %tmp21 = add nsw i32 %tmp16, %tmp14
  %tmp22 = getelementptr inbounds i32, ptr addrspace(1) %arg, i64 %tmp12
  store i32 %tmp21, ptr addrspace(1) %tmp22, align 4
  %tmp23 = add nuw nsw i64 %tmp11, 1
  %tmp24 = icmp slt i64 %tmp23, %tmp8
  br i1 %tmp24, label %bb10, label %bb26

bb26:                                             ; preds = %bb10, %bb20, %bb
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
