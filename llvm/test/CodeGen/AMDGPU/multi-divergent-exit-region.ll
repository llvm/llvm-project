; RUN: opt -mtriple=amdgcn-- -S -amdgpu-unify-divergent-exit-nodes -verify -structurizecfg -verify -si-annotate-control-flow %s | FileCheck -check-prefix=IR %s
; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; Add an extra verifier runs. There were some cases where invalid IR
; was produced but happened to be fixed by the later passes.

; Make sure divergent control flow with multiple exits from a region
; is properly handled. UnifyFunctionExitNodes should be run before
; StructurizeCFG.

; IR-LABEL: @multi_divergent_region_exit_ret_ret(
; IR: %1 = call { i1, i64 } @llvm.amdgcn.if.i64(i1 %0)
; IR: %2 = extractvalue { i1, i64 } %1, 0
; IR: %3 = extractvalue { i1, i64 } %1, 1
; IR: br i1 %2, label %LeafBlock1, label %Flow

; IR: Flow:
; IR: %4 = phi i1 [ true, %LeafBlock1 ], [ false, %entry ]
; IR: %5 = phi i1 [ %10, %LeafBlock1 ], [ false, %entry ]
; IR: %6 = call { i1, i64 } @llvm.amdgcn.else.i64.i64(i64 %3)
; IR: %7 = extractvalue { i1, i64 } %6, 0
; IR: %8 = extractvalue { i1, i64 } %6, 1
; IR: br i1 %7, label %LeafBlock, label %Flow1

; IR: LeafBlock:
; IR: br label %Flow1

; IR: LeafBlock1:
; IR: br label %Flow{{$}}

; IR:  Flow2:
; IR: %11 = phi i1 [ false, %exit1 ], [ %15, %Flow1 ]
; IR: call void @llvm.amdgcn.end.cf.i64(i64 %19)
; IR: %12 = call { i1, i64 } @llvm.amdgcn.if.i64(i1 %11)
; IR: %13 = extractvalue { i1, i64 } %12, 0
; IR: %14 = extractvalue { i1, i64 } %12, 1
; IR: br i1 %13, label %exit0, label %UnifiedReturnBlock

; IR: exit0:
; IR: store volatile i32 9, i32 addrspace(1)* undef
; IR: br label %UnifiedReturnBlock

; IR: Flow1:
; IR: %15 = phi i1 [ %SwitchLeaf, %LeafBlock ], [ %4, %Flow ]
; IR: %16 = phi i1 [ %9, %LeafBlock ], [ %5, %Flow ]
; IR: call void @llvm.amdgcn.end.cf.i64(i64 %8)
; IR: %17 = call { i1, i64 } @llvm.amdgcn.if.i64(i1 %16)
; IR: %18 = extractvalue { i1, i64 } %17, 0
; IR: %19 = extractvalue { i1, i64 } %17, 1
; IR: br i1 %18, label %exit1, label %Flow2

; IR: exit1:
; IR: store volatile i32 17, i32 addrspace(3)* undef
; IR:  br label %Flow2

; IR: UnifiedReturnBlock:
; IR: call void @llvm.amdgcn.end.cf.i64(i64 %14)
; IR: ret void


; GCN-LABEL: {{^}}multi_divergent_region_exit_ret_ret:

; GCN-DAG:  s_mov_b64           [[EXIT1:s\[[0-9]+:[0-9]+\]]], 0
; GCN-DAG:  v_cmp_lt_i32_e32    vcc, 1,
; GCN-DAG:  s_mov_b64           [[EXIT0:s\[[0-9]+:[0-9]+\]]], 0
; GCN-DAG:  s_and_saveexec_b64
; GCN-DAG:  s_xor_b64

; GCN: ; %LeafBlock1
; GCN-NEXT: s_mov_b64           [[EXIT0]], exec
; GCN-NEXT: v_cmp_ne_u32_e32    vcc, 2,
; GCN-NEXT: s_and_b64           [[EXIT1]], vcc, exec

; GCN: ; %Flow
; GCN-NEXT: s_or_saveexec_b64
; GCN-NEXT: s_xor_b64

; FIXME: Why is this compare essentially repeated?
; GCN: ; %LeafBlock
; GCN-DAG:  v_cmp_eq_u32_e32    vcc, 1,
; GCN-DAG:  v_cmp_ne_u32_e64    [[TMP1:s\[[0-9]+:[0-9]+\]]], 1,
; GCN-DAG:  s_andn2_b64         [[EXIT0]], [[EXIT0]], exec
; GCN-DAG:  s_andn2_b64         [[EXIT1]], [[EXIT1]], exec
; GCN-DAG:  s_and_b64           [[TMP0:s\[[0-9]+:[0-9]+\]]], vcc, exec
; GCN-DAG:  s_and_b64           [[TMP1]], [[TMP1]], exec
; GCN-DAG:  s_or_b64            [[EXIT0]], [[EXIT0]], [[TMP0]]
; GCN-DAG:  s_or_b64            [[EXIT1]], [[EXIT1]], [[TMP1]]

; GCN: ; %Flow4
; GCN-NEXT: s_or_b64            exec, exec,
; GCN-NEXT: s_and_saveexec_b64  {{s\[[0-9]+:[0-9]+\]}}, [[EXIT1]]
; GCN-NEXT: s_xor_b64

; GCN: ; %exit1
; GCN-DAG:  ds_write_b32
; GCN-DAG:  s_andn2_b64         [[EXIT0]], [[EXIT0]], exec

; GCN: ; %Flow5
; GCN-NEXT: s_or_b64            exec, exec,
; GCN-NEXT; s_and_saveexec_b64  {{s\[[0-9]+:[0-9]+\]}}, [[EXIT0]]

; GCN: ; %exit0
; GCN:      buffer_store_dword

; GCN: ; %UnifiedReturnBlock
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @multi_divergent_region_exit_ret_ret(i32 addrspace(1)* nocapture %arg0, i32 addrspace(1)* nocapture %arg1, i32 addrspace(1)* nocapture %arg2) #0 {
entry:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %tmp1 = add i32 0, %tmp
  %tmp2 = zext i32 %tmp1 to i64
  %tmp3 = add i64 0, %tmp2
  %tmp4 = shl i64 %tmp3, 32
  %tmp5 = ashr exact i64 %tmp4, 32
  %tmp6 = getelementptr inbounds i32, i32 addrspace(1)* %arg0, i64 %tmp5
  %tmp7 = load i32, i32 addrspace(1)* %tmp6, align 4
  %tmp8 = sext i32 %tmp7 to i64
  %tmp9 = getelementptr inbounds i32, i32 addrspace(1)* %arg1, i64 %tmp8
  %tmp10 = load i32, i32 addrspace(1)* %tmp9, align 4
  %tmp13 = zext i32 %tmp10 to i64
  %tmp14 = getelementptr inbounds i32, i32 addrspace(1)* %arg2, i64 %tmp13
  %tmp16 = load i32, i32 addrspace(1)* %tmp14, align 16
  %Pivot = icmp slt i32 %tmp16, 2
  br i1 %Pivot, label %LeafBlock, label %LeafBlock1

LeafBlock:                                        ; preds = %entry
  %SwitchLeaf = icmp eq i32 %tmp16, 1
  br i1 %SwitchLeaf, label %exit0, label %exit1

LeafBlock1:                                       ; preds = %entry
  %SwitchLeaf2 = icmp eq i32 %tmp16, 2
  br i1 %SwitchLeaf2, label %exit0, label %exit1

exit0:                                     ; preds = %LeafBlock, %LeafBlock1
  store volatile i32 9, i32 addrspace(1)* undef
  ret void

exit1:                                     ; preds = %LeafBlock, %LeafBlock1
  store volatile i32 17, i32 addrspace(3)* undef
  ret void
}

; IR-LABEL: @multi_divergent_region_exit_unreachable_unreachable(
; IR: %1 = call { i1, i64 } @llvm.amdgcn.if.i64(i1 %0)

; IR: %6 = call { i1, i64 } @llvm.amdgcn.else.i64.i64(i64 %3)

; IR: %11 = phi i1 [ false, %exit1 ], [ %15, %Flow1 ]
; IR: call void @llvm.amdgcn.end.cf.i64(i64 %19)
; IR: %12 = call { i1, i64 } @llvm.amdgcn.if.i64(i1 %11)
; IR: br i1 %13, label %exit0, label %UnifiedUnreachableBlock


; IR: UnifiedUnreachableBlock:
; IR-NEXT: unreachable


; FIXME: Probably should insert an s_endpgm anyway.
; GCN-LABEL: {{^}}multi_divergent_region_exit_unreachable_unreachable:
; GCN: ; %UnifiedUnreachableBlock
; GCN-NEXT: .Lfunc_end
define amdgpu_kernel void @multi_divergent_region_exit_unreachable_unreachable(i32 addrspace(1)* nocapture %arg0, i32 addrspace(1)* nocapture %arg1, i32 addrspace(1)* nocapture %arg2) #0 {
entry:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %tmp1 = add i32 0, %tmp
  %tmp2 = zext i32 %tmp1 to i64
  %tmp3 = add i64 0, %tmp2
  %tmp4 = shl i64 %tmp3, 32
  %tmp5 = ashr exact i64 %tmp4, 32
  %tmp6 = getelementptr inbounds i32, i32 addrspace(1)* %arg0, i64 %tmp5
  %tmp7 = load i32, i32 addrspace(1)* %tmp6, align 4
  %tmp8 = sext i32 %tmp7 to i64
  %tmp9 = getelementptr inbounds i32, i32 addrspace(1)* %arg1, i64 %tmp8
  %tmp10 = load i32, i32 addrspace(1)* %tmp9, align 4
  %tmp13 = zext i32 %tmp10 to i64
  %tmp14 = getelementptr inbounds i32, i32 addrspace(1)* %arg2, i64 %tmp13
  %tmp16 = load i32, i32 addrspace(1)* %tmp14, align 16
  %Pivot = icmp slt i32 %tmp16, 2
  br i1 %Pivot, label %LeafBlock, label %LeafBlock1

LeafBlock:                                        ; preds = %entry
  %SwitchLeaf = icmp eq i32 %tmp16, 1
  br i1 %SwitchLeaf, label %exit0, label %exit1

LeafBlock1:                                       ; preds = %entry
  %SwitchLeaf2 = icmp eq i32 %tmp16, 2
  br i1 %SwitchLeaf2, label %exit0, label %exit1

exit0:                                     ; preds = %LeafBlock, %LeafBlock1
  store volatile i32 9, i32 addrspace(1)* undef
  unreachable

exit1:                                     ; preds = %LeafBlock, %LeafBlock1
  store volatile i32 17, i32 addrspace(3)* undef
  unreachable
}

; IR-LABEL: @multi_exit_region_divergent_ret_uniform_ret(
; IR: %divergent.cond0 = icmp slt i32 %tmp16, 2
; IR: llvm.amdgcn.if
; IR: br i1

; IR: {{^}}Flow:
; IR: %4 = phi i1 [ true, %LeafBlock1 ], [ false, %entry ]
; IR: %5 = phi i1 [ %10, %LeafBlock1 ], [ false, %entry ]
; IR: %6 = call { i1, i64 } @llvm.amdgcn.else.i64.i64(i64 %3)
; IR: br i1 %7, label %LeafBlock, label %Flow1

; IR: {{^}}LeafBlock:
; IR: %divergent.cond1 = icmp eq i32 %tmp16, 1
; IR: %9 = xor i1 %divergent.cond1, true
; IR: br label %Flow1

; IR: LeafBlock1:
; IR: %uniform.cond0 = icmp eq i32 %arg3, 2
; IR: %10 = xor i1 %uniform.cond0, true
; IR: br label %Flow

; IR: Flow2:
; IR: %11 = phi i1 [ false, %exit1 ], [ %15, %Flow1 ]
; IR: call void @llvm.amdgcn.end.cf.i64(i64 %19)
; IR: %12 = call { i1, i64 } @llvm.amdgcn.if.i64(i1 %11)
; IR: br i1 %13, label %exit0, label %UnifiedReturnBlock

; IR: exit0:
; IR: store volatile i32 9, i32 addrspace(1)* undef
; IR: br label %UnifiedReturnBlock

; IR: {{^}}Flow1:
; IR: %15 = phi i1 [ %divergent.cond1, %LeafBlock ], [ %4, %Flow ]
; IR: %16 = phi i1 [ %9, %LeafBlock ], [ %5, %Flow ]
; IR: call void @llvm.amdgcn.end.cf.i64(i64 %8)
; IR: %17 = call { i1, i64 } @llvm.amdgcn.if.i64(i1 %16)
; IR: %18 = extractvalue { i1, i64 } %17, 0
; IR: %19 = extractvalue { i1, i64 } %17, 1
; IR: br i1 %18, label %exit1, label %Flow2

; IR: exit1:
; IR: store volatile i32 17, i32 addrspace(3)* undef
; IR: br label %Flow2

; IR: UnifiedReturnBlock:
; IR: call void @llvm.amdgcn.end.cf.i64(i64 %14)
; IR: ret void
define amdgpu_kernel void @multi_exit_region_divergent_ret_uniform_ret(i32 addrspace(1)* nocapture %arg0, i32 addrspace(1)* nocapture %arg1, i32 addrspace(1)* nocapture %arg2, i32 %arg3) #0 {
entry:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %tmp1 = add i32 0, %tmp
  %tmp2 = zext i32 %tmp1 to i64
  %tmp3 = add i64 0, %tmp2
  %tmp4 = shl i64 %tmp3, 32
  %tmp5 = ashr exact i64 %tmp4, 32
  %tmp6 = getelementptr inbounds i32, i32 addrspace(1)* %arg0, i64 %tmp5
  %tmp7 = load i32, i32 addrspace(1)* %tmp6, align 4
  %tmp8 = sext i32 %tmp7 to i64
  %tmp9 = getelementptr inbounds i32, i32 addrspace(1)* %arg1, i64 %tmp8
  %tmp10 = load i32, i32 addrspace(1)* %tmp9, align 4
  %tmp13 = zext i32 %tmp10 to i64
  %tmp14 = getelementptr inbounds i32, i32 addrspace(1)* %arg2, i64 %tmp13
  %tmp16 = load i32, i32 addrspace(1)* %tmp14, align 16
  %divergent.cond0 = icmp slt i32 %tmp16, 2
  br i1 %divergent.cond0, label %LeafBlock, label %LeafBlock1

LeafBlock:                                        ; preds = %entry
  %divergent.cond1 = icmp eq i32 %tmp16, 1
  br i1 %divergent.cond1, label %exit0, label %exit1

LeafBlock1:                                       ; preds = %entry
  %uniform.cond0 = icmp eq i32 %arg3, 2
  br i1 %uniform.cond0, label %exit0, label %exit1

exit0:                                     ; preds = %LeafBlock, %LeafBlock1
  store volatile i32 9, i32 addrspace(1)* undef
  ret void

exit1:                                     ; preds = %LeafBlock, %LeafBlock1
  store volatile i32 17, i32 addrspace(3)* undef
  ret void
}

; IR-LABEL: @multi_exit_region_uniform_ret_divergent_ret(
; IR: %1 = call { i1, i64 } @llvm.amdgcn.if.i64(i1 %0)
; IR: br i1 %2, label %LeafBlock1, label %Flow

; IR: Flow:
; IR: %4 = phi i1 [ true, %LeafBlock1 ], [ false, %entry ]
; IR: %5 = phi i1 [ %10, %LeafBlock1 ], [ false, %entry ]
; IR: %6 = call { i1, i64 } @llvm.amdgcn.else.i64.i64(i64 %3)

; IR: %11 = phi i1 [ false, %exit1 ], [ %15, %Flow1 ]
; IR: call void @llvm.amdgcn.end.cf.i64(i64 %19)
; IR: %12 = call { i1, i64 } @llvm.amdgcn.if.i64(i1 %11)

define amdgpu_kernel void @multi_exit_region_uniform_ret_divergent_ret(i32 addrspace(1)* nocapture %arg0, i32 addrspace(1)* nocapture %arg1, i32 addrspace(1)* nocapture %arg2, i32 %arg3) #0 {
entry:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %tmp1 = add i32 0, %tmp
  %tmp2 = zext i32 %tmp1 to i64
  %tmp3 = add i64 0, %tmp2
  %tmp4 = shl i64 %tmp3, 32
  %tmp5 = ashr exact i64 %tmp4, 32
  %tmp6 = getelementptr inbounds i32, i32 addrspace(1)* %arg0, i64 %tmp5
  %tmp7 = load i32, i32 addrspace(1)* %tmp6, align 4
  %tmp8 = sext i32 %tmp7 to i64
  %tmp9 = getelementptr inbounds i32, i32 addrspace(1)* %arg1, i64 %tmp8
  %tmp10 = load i32, i32 addrspace(1)* %tmp9, align 4
  %tmp13 = zext i32 %tmp10 to i64
  %tmp14 = getelementptr inbounds i32, i32 addrspace(1)* %arg2, i64 %tmp13
  %tmp16 = load i32, i32 addrspace(1)* %tmp14, align 16
  %Pivot = icmp slt i32 %tmp16, 2
  br i1 %Pivot, label %LeafBlock, label %LeafBlock1

LeafBlock:                                        ; preds = %entry
  %SwitchLeaf = icmp eq i32 %arg3, 1
  br i1 %SwitchLeaf, label %exit0, label %exit1

LeafBlock1:                                       ; preds = %entry
  %SwitchLeaf2 = icmp eq i32 %tmp16, 2
  br i1 %SwitchLeaf2, label %exit0, label %exit1

exit0:                                     ; preds = %LeafBlock, %LeafBlock1
  store volatile i32 9, i32 addrspace(1)* undef
  ret void

exit1:                                     ; preds = %LeafBlock, %LeafBlock1
  store volatile i32 17, i32 addrspace(3)* undef
  ret void
}

; IR-LABEL: @multi_divergent_region_exit_ret_ret_return_value(
; IR: Flow2:
; IR: %11 = phi i1 [ false, %exit1 ], [ %15, %Flow1 ]
; IR: call void @llvm.amdgcn.end.cf.i64(i64 %19)

; IR: UnifiedReturnBlock:
; IR: %UnifiedRetVal = phi float [ 2.000000e+00, %Flow2 ], [ 1.000000e+00, %exit0 ]
; IR: call void @llvm.amdgcn.end.cf.i64(i64 %14)
; IR: ret float %UnifiedRetVal
define amdgpu_ps float @multi_divergent_region_exit_ret_ret_return_value(i32 %vgpr) #0 {
entry:
  %Pivot = icmp slt i32 %vgpr, 2
  br i1 %Pivot, label %LeafBlock, label %LeafBlock1

LeafBlock:                                        ; preds = %entry
  %SwitchLeaf = icmp eq i32 %vgpr, 1
  br i1 %SwitchLeaf, label %exit0, label %exit1

LeafBlock1:                                       ; preds = %entry
  %SwitchLeaf2 = icmp eq i32 %vgpr, 2
  br i1 %SwitchLeaf2, label %exit0, label %exit1

exit0:                                     ; preds = %LeafBlock, %LeafBlock1
  store i32 9, i32 addrspace(1)* undef
  ret float 1.0

exit1:                                     ; preds = %LeafBlock, %LeafBlock1
  store i32 17, i32 addrspace(3)* undef
  ret float 2.0
}

; IR-LABEL: @uniform_branch_to_multi_divergent_region_exit_ret_ret_return_value(

; GCN-LABEL: {{^}}uniform_branch_to_multi_divergent_region_exit_ret_ret_return_value:
; GCN: s_cmp_gt_i32 s0, 1
; GCN: s_cbranch_scc0 [[FLOW:BB[0-9]+_[0-9]+]]

; GCN: v_cmp_ne_u32_e32 vcc, 7, v0

; GCN: {{^}}[[FLOW]]:
; GCN: s_cbranch_vccnz [[FLOW1:BB[0-9]+]]

; GCN: s_or_b64 exec, exec
; GCN: v_mov_b32_e32 v0, 2.0
; GCN-NOT: s_and_b64 exec, exec
; GCN: v_mov_b32_e32 v0, 1.0

; GCN: {{^BB[0-9]+_[0-9]+}}: ; %UnifiedReturnBlock
; GCN-NEXT: s_or_b64 exec, exec
; GCN-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0)
; GCN-NEXT: ; return

define amdgpu_ps float @uniform_branch_to_multi_divergent_region_exit_ret_ret_return_value(i32 inreg %sgpr, i32 %vgpr) #0 {
entry:
  %uniform.cond = icmp slt i32 %sgpr, 2
  br i1 %uniform.cond, label %LeafBlock, label %LeafBlock1

LeafBlock:                                        ; preds = %entry
  %divergent.cond0 = icmp eq i32 %vgpr, 3
  br i1 %divergent.cond0, label %exit0, label %exit1

LeafBlock1:                                       ; preds = %entry
  %divergent.cond1 = icmp eq i32 %vgpr, 7
  br i1 %divergent.cond1, label %exit0, label %exit1

exit0:                                     ; preds = %LeafBlock, %LeafBlock1
  store i32 9, i32 addrspace(1)* undef
  ret float 1.0

exit1:                                     ; preds = %LeafBlock, %LeafBlock1
  store i32 17, i32 addrspace(3)* undef
  ret float 2.0
}

; IR-LABEL: @multi_divergent_region_exit_ret_unreachable(
; IR: %1 = call { i1, i64 } @llvm.amdgcn.if.i64(i1 %0)

; IR: Flow:
; IR: %4 = phi i1 [ true, %LeafBlock1 ], [ false, %entry ]
; IR: %5 = phi i1 [ %10, %LeafBlock1 ], [ false, %entry ]
; IR: %6 = call { i1, i64 } @llvm.amdgcn.else.i64.i64(i64 %3)

; IR: Flow2:
; IR: %11 = phi i1 [ false, %exit1 ], [ %15, %Flow1 ]
; IR: call void @llvm.amdgcn.end.cf.i64(i64 %19)
; IR: %12 = call { i1, i64 } @llvm.amdgcn.if.i64(i1 %11)
; IR: br i1 %13, label %exit0, label %UnifiedReturnBlock

; IR: exit0:
; IR-NEXT: store volatile i32 17, i32 addrspace(3)* undef
; IR-NEXT: br label %UnifiedReturnBlock

; IR: Flow1:
; IR: %15 = phi i1 [ %SwitchLeaf, %LeafBlock ], [ %4, %Flow ]
; IR: %16 = phi i1 [ %9, %LeafBlock ], [ %5, %Flow ]
; IR: call void @llvm.amdgcn.end.cf.i64(i64 %8)
; IR: %17 = call { i1, i64 } @llvm.amdgcn.if.i64(i1 %16)
; IR: %18 = extractvalue { i1, i64 } %17, 0
; IR: %19 = extractvalue { i1, i64 } %17, 1
; IR: br i1 %18, label %exit1, label %Flow2

; IR: exit1:
; IR-NEXT: store volatile i32 9, i32 addrspace(1)* undef
; IR-NEXT: call void @llvm.amdgcn.unreachable()
; IR-NEXT: br label %Flow2

; IR: UnifiedReturnBlock:
; IR-NEXT: call void @llvm.amdgcn.end.cf.i64(i64 %14)
; IR-NEXT: ret void
define amdgpu_kernel void @multi_divergent_region_exit_ret_unreachable(i32 addrspace(1)* nocapture %arg0, i32 addrspace(1)* nocapture %arg1, i32 addrspace(1)* nocapture %arg2) #0 {
entry:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %tmp1 = add i32 0, %tmp
  %tmp2 = zext i32 %tmp1 to i64
  %tmp3 = add i64 0, %tmp2
  %tmp4 = shl i64 %tmp3, 32
  %tmp5 = ashr exact i64 %tmp4, 32
  %tmp6 = getelementptr inbounds i32, i32 addrspace(1)* %arg0, i64 %tmp5
  %tmp7 = load i32, i32 addrspace(1)* %tmp6, align 4
  %tmp8 = sext i32 %tmp7 to i64
  %tmp9 = getelementptr inbounds i32, i32 addrspace(1)* %arg1, i64 %tmp8
  %tmp10 = load i32, i32 addrspace(1)* %tmp9, align 4
  %tmp13 = zext i32 %tmp10 to i64
  %tmp14 = getelementptr inbounds i32, i32 addrspace(1)* %arg2, i64 %tmp13
  %tmp16 = load i32, i32 addrspace(1)* %tmp14, align 16
  %Pivot = icmp slt i32 %tmp16, 2
  br i1 %Pivot, label %LeafBlock, label %LeafBlock1

LeafBlock:                                        ; preds = %entry
  %SwitchLeaf = icmp eq i32 %tmp16, 1
  br i1 %SwitchLeaf, label %exit0, label %exit1

LeafBlock1:                                       ; preds = %entry
  %SwitchLeaf2 = icmp eq i32 %tmp16, 2
  br i1 %SwitchLeaf2, label %exit0, label %exit1

exit0:                                     ; preds = %LeafBlock, %LeafBlock1
  store volatile i32 17, i32 addrspace(3)* undef
  ret void

exit1:                                     ; preds = %LeafBlock, %LeafBlock1
  store volatile i32 9, i32 addrspace(1)* undef
  unreachable
}

; The non-uniformity of the branch to the exiting blocks requires
; looking at transitive predecessors.

; IR-LABEL: @indirect_multi_divergent_region_exit_ret_unreachable(

; IR: exit0:                                            ; preds = %Flow2
; IR-NEXT: store volatile i32 17, i32 addrspace(3)* undef
; IR-NEXT: br label %UnifiedReturnBlock


; IR: indirect.exit1:
; IR: %load = load volatile i32, i32 addrspace(1)* undef
; IR: store volatile i32 %load, i32 addrspace(1)* undef
; IR: store volatile i32 9, i32 addrspace(1)* undef
; IR: call void @llvm.amdgcn.unreachable()
; IR-NEXT: br label %Flow2

; IR: UnifiedReturnBlock:                               ; preds = %exit0, %Flow2
; IR-NEXT: call void @llvm.amdgcn.end.cf.i64(i64 %14)
; IR-NEXT: ret void
define amdgpu_kernel void @indirect_multi_divergent_region_exit_ret_unreachable(i32 addrspace(1)* nocapture %arg0, i32 addrspace(1)* nocapture %arg1, i32 addrspace(1)* nocapture %arg2) #0 {
entry:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %tmp1 = add i32 0, %tmp
  %tmp2 = zext i32 %tmp1 to i64
  %tmp3 = add i64 0, %tmp2
  %tmp4 = shl i64 %tmp3, 32
  %tmp5 = ashr exact i64 %tmp4, 32
  %tmp6 = getelementptr inbounds i32, i32 addrspace(1)* %arg0, i64 %tmp5
  %tmp7 = load i32, i32 addrspace(1)* %tmp6, align 4
  %tmp8 = sext i32 %tmp7 to i64
  %tmp9 = getelementptr inbounds i32, i32 addrspace(1)* %arg1, i64 %tmp8
  %tmp10 = load i32, i32 addrspace(1)* %tmp9, align 4
  %tmp13 = zext i32 %tmp10 to i64
  %tmp14 = getelementptr inbounds i32, i32 addrspace(1)* %arg2, i64 %tmp13
  %tmp16 = load i32, i32 addrspace(1)* %tmp14, align 16
  %Pivot = icmp slt i32 %tmp16, 2
  br i1 %Pivot, label %LeafBlock, label %LeafBlock1

LeafBlock:                                        ; preds = %entry
  %SwitchLeaf = icmp eq i32 %tmp16, 1
  br i1 %SwitchLeaf, label %exit0, label %indirect.exit1

LeafBlock1:                                       ; preds = %entry
  %SwitchLeaf2 = icmp eq i32 %tmp16, 2
  br i1 %SwitchLeaf2, label %exit0, label %indirect.exit1

exit0:                                     ; preds = %LeafBlock, %LeafBlock1
  store volatile i32 17, i32 addrspace(3)* undef
  ret void

indirect.exit1:
  %load = load volatile i32, i32 addrspace(1)* undef
  store volatile i32 %load, i32 addrspace(1)* undef
  br label %exit1

exit1:                                     ; preds = %LeafBlock, %LeafBlock1
  store volatile i32 9, i32 addrspace(1)* undef
  unreachable
}

; IR-LABEL: @multi_divergent_region_exit_ret_switch(
define amdgpu_kernel void @multi_divergent_region_exit_ret_switch(i32 addrspace(1)* nocapture %arg0, i32 addrspace(1)* nocapture %arg1, i32 addrspace(1)* nocapture %arg2) #0 {
entry:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %tmp1 = add i32 0, %tmp
  %tmp2 = zext i32 %tmp1 to i64
  %tmp3 = add i64 0, %tmp2
  %tmp4 = shl i64 %tmp3, 32
  %tmp5 = ashr exact i64 %tmp4, 32
  %tmp6 = getelementptr inbounds i32, i32 addrspace(1)* %arg0, i64 %tmp5
  %tmp7 = load i32, i32 addrspace(1)* %tmp6, align 4
  %tmp8 = sext i32 %tmp7 to i64
  %tmp9 = getelementptr inbounds i32, i32 addrspace(1)* %arg1, i64 %tmp8
  %tmp10 = load i32, i32 addrspace(1)* %tmp9, align 4
  %tmp13 = zext i32 %tmp10 to i64
  %tmp14 = getelementptr inbounds i32, i32 addrspace(1)* %arg2, i64 %tmp13
  %tmp16 = load i32, i32 addrspace(1)* %tmp14, align 16
  switch i32 %tmp16, label %exit1
    [ i32 1, label %LeafBlock
      i32 2, label %LeafBlock1
      i32 3, label %exit0 ]

LeafBlock:                                        ; preds = %entry
  %SwitchLeaf = icmp eq i32 %tmp16, 1
  br i1 %SwitchLeaf, label %exit0, label %exit1

LeafBlock1:                                       ; preds = %entry
  %SwitchLeaf2 = icmp eq i32 %tmp16, 2
  br i1 %SwitchLeaf2, label %exit0, label %exit1

exit0:                                     ; preds = %LeafBlock, %LeafBlock1
  store volatile i32 17, i32 addrspace(3)* undef
  ret void

exit1:                                     ; preds = %LeafBlock, %LeafBlock1
  store volatile i32 9, i32 addrspace(1)* undef
  unreachable
}

; IR-LABEL: @divergent_multi_ret_nest_in_uniform_triangle(
define amdgpu_kernel void @divergent_multi_ret_nest_in_uniform_triangle(i32 %arg0) #0 {
entry:
  %uniform.cond0 = icmp eq i32 %arg0, 4
  br i1 %uniform.cond0, label %divergent.multi.exit.region, label %uniform.ret

divergent.multi.exit.region:
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %divergent.cond0 = icmp eq i32 %id.x, 0
  br i1 %divergent.cond0, label %divergent.ret0, label %divergent.ret1

divergent.ret0:
  store volatile i32 11, i32 addrspace(3)* undef
  ret void

divergent.ret1:
  store volatile i32 42, i32 addrspace(3)* undef
  ret void

uniform.ret:
  store volatile i32 9, i32 addrspace(1)* undef
  ret void
}

; IR-LABEL: @divergent_complex_multi_ret_nest_in_uniform_triangle(
define amdgpu_kernel void @divergent_complex_multi_ret_nest_in_uniform_triangle(i32 %arg0) #0 {
entry:
  %uniform.cond0 = icmp eq i32 %arg0, 4
  br i1 %uniform.cond0, label %divergent.multi.exit.region, label %uniform.ret

divergent.multi.exit.region:
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %divergent.cond0 = icmp eq i32 %id.x, 0
  br i1 %divergent.cond0, label %divergent.if, label %divergent.ret1

divergent.if:
  %vgpr0 = load volatile float, float addrspace(1)* undef
  %divergent.cond1 = fcmp ogt float %vgpr0, 1.0
  br i1 %divergent.cond1, label %divergent.then, label %divergent.endif

divergent.then:
  %vgpr1 = load volatile float, float addrspace(1)* undef
  %divergent.cond2 = fcmp olt float %vgpr1, 4.0
  store volatile i32 33, i32 addrspace(1)* undef
  br i1 %divergent.cond2, label %divergent.ret0, label %divergent.endif

divergent.endif:
  store volatile i32 38, i32 addrspace(1)* undef
  br label %divergent.ret0

divergent.ret0:
  store volatile i32 11, i32 addrspace(3)* undef
  ret void

divergent.ret1:
  store volatile i32 42, i32 addrspace(3)* undef
  ret void

uniform.ret:
  store volatile i32 9, i32 addrspace(1)* undef
  ret void
}

; IR-LABEL: @uniform_complex_multi_ret_nest_in_divergent_triangle(
; IR: Flow1:                                            ; preds = %uniform.ret1, %uniform.multi.exit.region
; IR: %8 = phi i1 [ false, %uniform.ret1 ], [ true, %uniform.multi.exit.region ]
; IR: br i1 %8, label %uniform.if, label %Flow2

; IR: Flow:                                             ; preds = %uniform.then, %uniform.if
; IR: %11 = phi i1 [ %10, %uniform.then ], [ %9, %uniform.if ]
; IR: br i1 %11, label %uniform.endif, label %uniform.ret0

; IR: UnifiedReturnBlock:                               ; preds = %Flow3, %Flow2
; IR-NEXT: call void @llvm.amdgcn.end.cf.i64(i64 %6)
; IR-NEXT: ret void
define amdgpu_kernel void @uniform_complex_multi_ret_nest_in_divergent_triangle(i32 %arg0) #0 {
entry:
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %divergent.cond0 = icmp eq i32 %id.x, 0
  br i1 %divergent.cond0, label %uniform.multi.exit.region, label %divergent.ret

uniform.multi.exit.region:
  %uniform.cond0 = icmp eq i32 %arg0, 4
  br i1 %uniform.cond0, label %uniform.if, label %uniform.ret1

uniform.if:
  %sgpr0 = load volatile i32, i32 addrspace(4)* undef
  %uniform.cond1 = icmp slt i32 %sgpr0, 1
  br i1 %uniform.cond1, label %uniform.then, label %uniform.endif

uniform.then:
  %sgpr1 = load volatile i32, i32 addrspace(4)* undef
  %uniform.cond2 = icmp sge i32 %sgpr1, 4
  store volatile i32 33, i32 addrspace(1)* undef
  br i1 %uniform.cond2, label %uniform.ret0, label %uniform.endif

uniform.endif:
  store volatile i32 38, i32 addrspace(1)* undef
  br label %uniform.ret0

uniform.ret0:
  store volatile i32 11, i32 addrspace(3)* undef
  ret void

uniform.ret1:
  store volatile i32 42, i32 addrspace(3)* undef
  ret void

divergent.ret:
  store volatile i32 9, i32 addrspace(1)* undef
  ret void
}

; IR-LABEL: @multi_divergent_unreachable_exit(
; IR: UnifiedUnreachableBlock:
; IR-NEXT: call void @llvm.amdgcn.unreachable()
; IR-NEXT: br label %UnifiedReturnBlock

; IR: UnifiedReturnBlock:
; IR-NEXT: call void @llvm.amdgcn.end.cf.i64(i64
; IR-NEXT: ret void
define amdgpu_kernel void @multi_divergent_unreachable_exit() #0 {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  switch i32 %tmp, label %bb3 [
    i32 2, label %bb1
    i32 0, label %bb2
  ]

bb1:                                              ; preds = %bb
  unreachable

bb2:                                              ; preds = %bb
  unreachable

bb3:                                              ; preds = %bb
  switch i32 undef, label %bb5 [
    i32 2, label %bb4
  ]

bb4:                                              ; preds = %bb3
  ret void

bb5:                                              ; preds = %bb3
  unreachable
}

; Test that there is an extra export inserted after the normal export,
; if the normal export is inside a uniformly reached block and there is
; an infinite loop in the pixel shader.

; IR-LABEL: @uniformly_reached_export
; IR-NEXT: .entry:
; IR: br i1 [[CND:%.*]], label %[[EXP:.*]], label %[[FLOW:.*]]

; IR: [[FLOW]]:
; IR-NEXT: phi
; IR-NEXT: br i1 [[CND2:%.*]], label %[[LOOP:.*]], label %UnifiedReturnBlock

; IR: [[LOOP]]:
; IR-NEXT: br i1 false, label %[[FLOW1:.*]], label %[[LOOP]]

; IR: [[EXP]]:
; IR-NEXT: call void @llvm.amdgcn.exp.compr.v2f16(i32 immarg 0, i32 immarg 15, <2 x half> <half 0xH3C00, half 0xH0000>, <2 x half> <half 0xH0000, half 0xH3C00>, i1 immarg false, i1 immarg true)
; IR-NEXT: br label %[[FLOW]]

; IR: [[FLOW1]]:
; IR-NEXT: br label %UnifiedReturnBlock

; IR: UnifiedReturnBlock:
; IR-NEXT: call void @llvm.amdgcn.exp.f32(i32 9, i32 0, float undef, float undef, float undef, float undef, i1 true, i1 true)
; IR-NEXT: ret void

define amdgpu_ps void @uniformly_reached_export(float inreg %tmp25) {
.entry:
  %tmp26 = fcmp olt float %tmp25, 0.000000e+00
  br i1 %tmp26, label %loop, label %bb27

loop:                                               ; preds = %loop, %.entry
  br label %loop

bb27:                                             ; preds = %.entry
  call void @llvm.amdgcn.exp.compr.v2f16(i32 immarg 0, i32 immarg 15, <2 x half> <half 0xH3C00, half 0xH0000>, <2 x half> <half 0xH0000, half 0xH3C00>, i1 immarg true, i1 immarg true)
  ret void
}

declare void @llvm.amdgcn.exp.compr.v2f16(i32 immarg, i32 immarg, <2 x half>, <2 x half>, i1 immarg, i1 immarg) #0
declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
