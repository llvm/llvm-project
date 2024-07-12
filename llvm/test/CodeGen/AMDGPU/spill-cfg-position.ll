; RUN: llc -amdgpu-scalarize-global-loads=false -mtriple=amdgcn -mcpu=tahiti -verify-machineinstrs -stress-regalloc=6 < %s | FileCheck %s

; Inline spiller can decide to move a spill as early as possible in the basic block.
; It will skip phis and label, but we also need to make sure it skips instructions
; in the basic block prologue which restore exec mask.
; Make sure instruction to restore exec mask immediately follows label

; CHECK-LABEL: {{^}}spill_cfg_position:
; CHECK: s_cbranch_execz [[LABEL1:.LBB[0-9_]+]]
; CHECK: {{^}}[[LABEL1]]:
; CHECK: s_cbranch_execz [[LABEL2:.LBB[0-9_]+]]
; CHECK: {{^}}[[LABEL2]]:
; CHECK-NEXT: s_or_b64 exec
; CHECK: buffer_

define amdgpu_kernel void @spill_cfg_position(ptr addrspace(1) nocapture %arg) {
bb:
  %tmp1 = tail call i32 @llvm.amdgcn.workitem.id.x() #0
  %tmp14 = load i32, ptr addrspace(1) %arg, align 4
  %tmp15 = getelementptr inbounds i32, ptr addrspace(1) %arg, i64 1
  %tmp16 = load i32, ptr addrspace(1) %tmp15, align 4
  %tmp17 = getelementptr inbounds i32, ptr addrspace(1) %arg, i64 2
  %tmp18 = load i32, ptr addrspace(1) %tmp17, align 4
  %tmp19 = getelementptr inbounds i32, ptr addrspace(1) %arg, i64 3
  %tmp20 = load i32, ptr addrspace(1) %tmp19, align 4
  %tmp21 = getelementptr inbounds i32, ptr addrspace(1) %arg, i64 4
  %tmp22 = load i32, ptr addrspace(1) %tmp21, align 4
  %tmp23 = getelementptr inbounds i32, ptr addrspace(1) %arg, i64 5
  %tmp24 = load i32, ptr addrspace(1) %tmp23, align 4
  %tmp25 = getelementptr inbounds i32, ptr addrspace(1) %arg, i64 6
  %tmp26 = load i32, ptr addrspace(1) %tmp25, align 4
  %tmp27 = getelementptr inbounds i32, ptr addrspace(1) %arg, i64 7
  %tmp28 = load i32, ptr addrspace(1) %tmp27, align 4
  %tmp29 = getelementptr inbounds i32, ptr addrspace(1) %arg, i64 8
  %tmp30 = load i32, ptr addrspace(1) %tmp29, align 4
  %tmp33 = getelementptr inbounds i32, ptr addrspace(1) %arg, i32 %tmp1
  %tmp34 = load i32, ptr addrspace(1) %tmp33, align 4
  %tmp35 = icmp eq i32 %tmp34, 0
  br i1 %tmp35, label %bb44, label %bb36

bb36:                                             ; preds = %bb
  %tmp37 = mul nsw i32 %tmp20, %tmp18
  %tmp38 = add nsw i32 %tmp37, %tmp16
  %tmp39 = mul nsw i32 %tmp24, %tmp22
  %tmp40 = add nsw i32 %tmp38, %tmp39
  %tmp41 = mul nsw i32 %tmp28, %tmp26
  %tmp42 = add nsw i32 %tmp40, %tmp41
  %tmp43 = add nsw i32 %tmp42, %tmp30
  br label %bb52

bb44:                                             ; preds = %bb
  %tmp45 = mul nsw i32 %tmp18, %tmp16
  %tmp46 = mul nsw i32 %tmp22, %tmp20
  %tmp47 = add nsw i32 %tmp46, %tmp45
  %tmp48 = mul nsw i32 %tmp26, %tmp24
  %tmp49 = add nsw i32 %tmp47, %tmp48
  %tmp50 = mul nsw i32 %tmp30, %tmp28
  %tmp51 = add nsw i32 %tmp49, %tmp50
  br label %bb52

bb52:                                             ; preds = %bb44, %bb36
  %tmp53 = phi i32 [ %tmp43, %bb36 ], [ %tmp51, %bb44 ]
  %tmp54 = mul nsw i32 %tmp16, %tmp14
  %tmp55 = mul nsw i32 %tmp22, %tmp18
  %tmp56 = mul nsw i32 %tmp24, %tmp20
  %tmp57 = mul nsw i32 %tmp30, %tmp26
  %tmp58 = add i32 %tmp55, %tmp54
  %tmp59 = add i32 %tmp58, %tmp56
  %tmp60 = add i32 %tmp59, %tmp28
  %tmp61 = add i32 %tmp60, %tmp57
  %tmp62 = add i32 %tmp61, %tmp53
  store i32 %tmp62, ptr addrspace(1) %tmp33, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone }
