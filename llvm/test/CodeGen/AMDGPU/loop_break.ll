; RUN: opt -mtriple=amdgcn-- -S -structurizecfg -si-annotate-control-flow %s | FileCheck -check-prefix=OPT %s
; RUN: llc -march=amdgcn -verify-machineinstrs -disable-block-placement < %s | FileCheck -check-prefix=GCN %s

; Uses llvm.amdgcn.break

; OPT-LABEL: @break_loop(
; OPT: bb1:
; OPT: icmp slt i32
; OPT-NEXT: br i1 %cmp0, label %bb4, label %Flow

; OPT: bb4:
; OPT: load volatile
; OPT: icmp slt i32
; OPT: xor i1 %cmp1
; OPT: br label %Flow

; OPT: Flow:
; OPT: call i64 @llvm.amdgcn.if.break.i64.i64(
; OPT: call i1 @llvm.amdgcn.loop.i64(i64
; OPT: br i1 %{{[0-9]+}}, label %bb9, label %bb1

; OPT: bb9:
; OPT: call void @llvm.amdgcn.end.cf.i64(i64

; GCN-LABEL: {{^}}break_loop:
; GCN:      s_mov_b64         [[ACCUM_MASK:s\[[0-9]+:[0-9]+\]]], 0{{$}}

; GCN: [[LOOP_ENTRY:BB[0-9]+_[0-9]+]]: ; %bb1
; GCN:     s_add_i32 s4, s4, 1
; GCN:     s_or_b64 [[INNER_MASK:s\[[0-9]+:[0-9]+\]]], [[INNER_MASK]], exec
; GCN:     s_cmp_gt_i32 s4, -1
; GCN:     s_cbranch_scc1   [[FLOW:BB[0-9]+_[0-9]+]]

; GCN: ; %bb4
; GCN:      buffer_load_dword
; GCN:      v_cmp_ge_i32_e32  vcc
; GCN:      s_andn2_b64 [[INNER_MASK]], [[INNER_MASK]], exec
; GCN:      s_and_b64 [[BROKEN_MASK:s\[[0-9]+:[0-9]+\]]], vcc, exec
; GCN:      s_or_b64  [[INNER_MASK]], [[INNER_MASK]], [[BROKEN_MASK]]

; GCN: [[FLOW]]: ; %Flow
; GCN:           ;   in Loop: Header=BB0_1 Depth=1
; GCN:      s_and_b64         [[AND_MASK:s\[[0-9]+:[0-9]+\]]], exec, [[INNER_MASK]]
; GCN-NEXT: s_or_b64          [[ACCUM_MASK]], [[AND_MASK]], [[ACCUM_MASK]]
; GCN-NEXT: s_andn2_b64       exec, exec, [[ACCUM_MASK]]
; GCN-NEXT: s_cbranch_execnz  [[LOOP_ENTRY]]

; GCN: ; %bb.4: ; %bb9
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @break_loop(i32 %arg) #0 {
bb:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %tmp = sub i32 %id, %arg
  br label %bb1

bb1:
  %lsr.iv = phi i32 [ undef, %bb ], [ %lsr.iv.next, %bb4 ]
  %lsr.iv.next = add i32 %lsr.iv, 1
  %cmp0 = icmp slt i32 %lsr.iv.next, 0
  br i1 %cmp0, label %bb4, label %bb9

bb4:
  %load = load volatile i32, i32 addrspace(1)* undef, align 4
  %cmp1 = icmp slt i32 %tmp, %load
  br i1 %cmp1, label %bb1, label %bb9

bb9:
  ret void
}

; OPT-LABEL: @undef_phi_cond_break_loop(
; OPT: bb1:
; OPT-NEXT: %phi.broken = phi i64 [ %0, %Flow ], [ 0, %bb ]
; OPT-NEXT: %lsr.iv = phi i32 [ undef, %bb ], [ %tmp2, %Flow ]
; OPT-NEXT: %lsr.iv.next = add i32 %lsr.iv, 1
; OPT-NEXT: %cmp0 = icmp slt i32 %lsr.iv.next, 0
; OPT-NEXT: br i1 %cmp0, label %bb4, label %Flow

; OPT: bb4:
; OPT-NEXT: %load = load volatile i32, i32 addrspace(1)* undef, align 4
; OPT-NEXT: %cmp1 = icmp sge i32 %tmp, %load
; OPT-NEXT: br label %Flow

; OPT: Flow:
; OPT-NEXT: %tmp2 = phi i32 [ %lsr.iv.next, %bb4 ], [ undef, %bb1 ]
; OPT-NEXT: %tmp3 = phi i1 [ %cmp1, %bb4 ], [ undef, %bb1 ]
; OPT-NEXT: %0 = call i64 @llvm.amdgcn.if.break.i64.i64(i1 %tmp3, i64 %phi.broken)
; OPT-NEXT: %1 = call i1 @llvm.amdgcn.loop.i64(i64 %0)
; OPT-NEXT: br i1 %1, label %bb9, label %bb1

; OPT: bb9:                                              ; preds = %Flow
; OPT-NEXT: call void @llvm.amdgcn.end.cf.i64(i64 %0)
; OPT-NEXT: store volatile i32 7
; OPT-NEXT: ret void
define amdgpu_kernel void @undef_phi_cond_break_loop(i32 %arg) #0 {
bb:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %tmp = sub i32 %id, %arg
  br label %bb1

bb1:                                              ; preds = %Flow, %bb
  %lsr.iv = phi i32 [ undef, %bb ], [ %tmp2, %Flow ]
  %lsr.iv.next = add i32 %lsr.iv, 1
  %cmp0 = icmp slt i32 %lsr.iv.next, 0
  br i1 %cmp0, label %bb4, label %Flow

bb4:                                              ; preds = %bb1
  %load = load volatile i32, i32 addrspace(1)* undef, align 4
  %cmp1 = icmp sge i32 %tmp, %load
  br label %Flow

Flow:                                             ; preds = %bb4, %bb1
  %tmp2 = phi i32 [ %lsr.iv.next, %bb4 ], [ undef, %bb1 ]
  %tmp3 = phi i1 [ %cmp1, %bb4 ], [ undef, %bb1 ]
  br i1 %tmp3, label %bb9, label %bb1

bb9:                                              ; preds = %Flow
  store volatile i32 7, i32 addrspace(3)* undef
  ret void
}

; FIXME: ConstantExpr compare of address to null folds away
@lds = addrspace(3) global i32 undef

; OPT-LABEL: @constexpr_phi_cond_break_loop(
; OPT: bb1:
; OPT-NEXT: %phi.broken = phi i64 [ %0, %Flow ], [ 0, %bb ]
; OPT-NEXT: %lsr.iv = phi i32 [ undef, %bb ], [ %tmp2, %Flow ]
; OPT-NEXT: %lsr.iv.next = add i32 %lsr.iv, 1
; OPT-NEXT: %cmp0 = icmp slt i32 %lsr.iv.next, 0
; OPT-NEXT: br i1 %cmp0, label %bb4, label %Flow

; OPT: bb4:
; OPT-NEXT: %load = load volatile i32, i32 addrspace(1)* undef, align 4
; OPT-NEXT: %cmp1 = icmp sge i32 %tmp, %load
; OPT-NEXT: br label %Flow

; OPT: Flow:
; OPT-NEXT: %tmp2 = phi i32 [ %lsr.iv.next, %bb4 ], [ undef, %bb1 ]
; OPT-NEXT: %tmp3 = phi i1 [ %cmp1, %bb4 ], [ icmp ne (i32 addrspace(3)* inttoptr (i32 4 to i32 addrspace(3)*), i32 addrspace(3)* @lds), %bb1 ]
; OPT-NEXT: %0 = call i64 @llvm.amdgcn.if.break.i64.i64(i1 %tmp3, i64 %phi.broken)
; OPT-NEXT: %1 = call i1 @llvm.amdgcn.loop.i64(i64 %0)
; OPT-NEXT: br i1 %1, label %bb9, label %bb1

; OPT: bb9:                                              ; preds = %Flow
; OPT-NEXT: call void @llvm.amdgcn.end.cf.i64(i64 %0)
; OPT-NEXT: store volatile i32 7
; OPT-NEXT: ret void
define amdgpu_kernel void @constexpr_phi_cond_break_loop(i32 %arg) #0 {
bb:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %tmp = sub i32 %id, %arg
  br label %bb1

bb1:                                              ; preds = %Flow, %bb
  %lsr.iv = phi i32 [ undef, %bb ], [ %tmp2, %Flow ]
  %lsr.iv.next = add i32 %lsr.iv, 1
  %cmp0 = icmp slt i32 %lsr.iv.next, 0
  br i1 %cmp0, label %bb4, label %Flow

bb4:                                              ; preds = %bb1
  %load = load volatile i32, i32 addrspace(1)* undef, align 4
  %cmp1 = icmp sge i32 %tmp, %load
  br label %Flow

Flow:                                             ; preds = %bb4, %bb1
  %tmp2 = phi i32 [ %lsr.iv.next, %bb4 ], [ undef, %bb1 ]
  %tmp3 = phi i1 [ %cmp1, %bb4 ], [ icmp ne (i32 addrspace(3)* inttoptr (i32 4 to i32 addrspace(3)*), i32 addrspace(3)* @lds), %bb1 ]
  br i1 %tmp3, label %bb9, label %bb1

bb9:                                              ; preds = %Flow
  store volatile i32 7, i32 addrspace(3)* undef
  ret void
}

; OPT-LABEL: @true_phi_cond_break_loop(
; OPT: bb1:
; OPT-NEXT: %phi.broken = phi i64 [ %0, %Flow ], [ 0, %bb ]
; OPT-NEXT: %lsr.iv = phi i32 [ undef, %bb ], [ %tmp2, %Flow ]
; OPT-NEXT: %lsr.iv.next = add i32 %lsr.iv, 1
; OPT-NEXT: %cmp0 = icmp slt i32 %lsr.iv.next, 0
; OPT-NEXT: br i1 %cmp0, label %bb4, label %Flow

; OPT: bb4:
; OPT-NEXT: %load = load volatile i32, i32 addrspace(1)* undef, align 4
; OPT-NEXT: %cmp1 = icmp sge i32 %tmp, %load
; OPT-NEXT: br label %Flow

; OPT: Flow:
; OPT-NEXT: %tmp2 = phi i32 [ %lsr.iv.next, %bb4 ], [ undef, %bb1 ]
; OPT-NEXT: %tmp3 = phi i1 [ %cmp1, %bb4 ], [ true, %bb1 ]
; OPT-NEXT: %0 = call i64 @llvm.amdgcn.if.break.i64.i64(i1 %tmp3, i64 %phi.broken)
; OPT-NEXT: %1 = call i1 @llvm.amdgcn.loop.i64(i64 %0)
; OPT-NEXT: br i1 %1, label %bb9, label %bb1

; OPT: bb9:                                              ; preds = %Flow
; OPT-NEXT: call void @llvm.amdgcn.end.cf.i64(i64 %0)
; OPT-NEXT: store volatile i32 7
; OPT-NEXT: ret void
define amdgpu_kernel void @true_phi_cond_break_loop(i32 %arg) #0 {
bb:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %tmp = sub i32 %id, %arg
  br label %bb1

bb1:                                              ; preds = %Flow, %bb
  %lsr.iv = phi i32 [ undef, %bb ], [ %tmp2, %Flow ]
  %lsr.iv.next = add i32 %lsr.iv, 1
  %cmp0 = icmp slt i32 %lsr.iv.next, 0
  br i1 %cmp0, label %bb4, label %Flow

bb4:                                              ; preds = %bb1
  %load = load volatile i32, i32 addrspace(1)* undef, align 4
  %cmp1 = icmp sge i32 %tmp, %load
  br label %Flow

Flow:                                             ; preds = %bb4, %bb1
  %tmp2 = phi i32 [ %lsr.iv.next, %bb4 ], [ undef, %bb1 ]
  %tmp3 = phi i1 [ %cmp1, %bb4 ], [ true, %bb1 ]
  br i1 %tmp3, label %bb9, label %bb1

bb9:                                              ; preds = %Flow
  store volatile i32 7, i32 addrspace(3)* undef
  ret void
}

; OPT-LABEL: @false_phi_cond_break_loop(
; OPT: bb1:
; OPT-NEXT: %phi.broken = phi i64 [ %0, %Flow ], [ 0, %bb ]
; OPT-NEXT: %lsr.iv = phi i32 [ undef, %bb ], [ %tmp2, %Flow ]
; OPT-NOT: call
; OPT: br i1 %cmp0, label %bb4, label %Flow

; OPT: bb4:
; OPT-NEXT: %load = load volatile i32, i32 addrspace(1)* undef, align 4
; OPT-NEXT: %cmp1 = icmp sge i32 %tmp, %load
; OPT-NEXT: br label %Flow

; OPT: Flow:
; OPT-NEXT: %tmp2 = phi i32 [ %lsr.iv.next, %bb4 ], [ undef, %bb1 ]
; OPT-NEXT: %tmp3 = phi i1 [ %cmp1, %bb4 ], [ false, %bb1 ]
; OPT-NEXT: %0 = call i64 @llvm.amdgcn.if.break.i64.i64(i1 %tmp3, i64 %phi.broken)
; OPT-NEXT: %1 = call i1 @llvm.amdgcn.loop.i64(i64 %0)
; OPT-NEXT: br i1 %1, label %bb9, label %bb1

; OPT: bb9:                                              ; preds = %Flow
; OPT-NEXT: call void @llvm.amdgcn.end.cf.i64(i64 %0)
; OPT-NEXT: store volatile i32 7
; OPT-NEXT: ret void
define amdgpu_kernel void @false_phi_cond_break_loop(i32 %arg) #0 {
bb:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %tmp = sub i32 %id, %arg
  br label %bb1

bb1:                                              ; preds = %Flow, %bb
  %lsr.iv = phi i32 [ undef, %bb ], [ %tmp2, %Flow ]
  %lsr.iv.next = add i32 %lsr.iv, 1
  %cmp0 = icmp slt i32 %lsr.iv.next, 0
  br i1 %cmp0, label %bb4, label %Flow

bb4:                                              ; preds = %bb1
  %load = load volatile i32, i32 addrspace(1)* undef, align 4
  %cmp1 = icmp sge i32 %tmp, %load
  br label %Flow

Flow:                                             ; preds = %bb4, %bb1
  %tmp2 = phi i32 [ %lsr.iv.next, %bb4 ], [ undef, %bb1 ]
  %tmp3 = phi i1 [ %cmp1, %bb4 ], [ false, %bb1 ]
  br i1 %tmp3, label %bb9, label %bb1

bb9:                                              ; preds = %Flow
  store volatile i32 7, i32 addrspace(3)* undef
  ret void
}

; Swap order of branches in flow block so that the true phi is
; continue.

; OPT-LABEL: @invert_true_phi_cond_break_loop(
; OPT: bb1:
; OPT-NEXT: %phi.broken = phi i64 [ %1, %Flow ], [ 0, %bb ]
; OPT-NEXT: %lsr.iv = phi i32 [ undef, %bb ], [ %tmp2, %Flow ]
; OPT-NEXT: %lsr.iv.next = add i32 %lsr.iv, 1
; OPT-NEXT: %cmp0 = icmp slt i32 %lsr.iv.next, 0
; OPT-NEXT: br i1 %cmp0, label %bb4, label %Flow

; OPT: bb4:
; OPT-NEXT: %load = load volatile i32, i32 addrspace(1)* undef, align 4
; OPT-NEXT: %cmp1 = icmp sge i32 %tmp, %load
; OPT-NEXT: br label %Flow

; OPT: Flow:
; OPT-NEXT: %tmp2 = phi i32 [ %lsr.iv.next, %bb4 ], [ undef, %bb1 ]
; OPT-NEXT: %tmp3 = phi i1 [ %cmp1, %bb4 ], [ true, %bb1 ]
; OPT-NEXT: %0 = xor i1 %tmp3, true
; OPT-NEXT: %1 = call i64 @llvm.amdgcn.if.break.i64.i64(i1 %0, i64 %phi.broken)
; OPT-NEXT: %2 = call i1 @llvm.amdgcn.loop.i64(i64 %1)
; OPT-NEXT: br i1 %2, label %bb9, label %bb1

; OPT: bb9:
; OPT-NEXT: call void @llvm.amdgcn.end.cf.i64(i64 %1)
; OPT-NEXT: store volatile i32 7, i32 addrspace(3)* undef
; OPT-NEXT: ret void
define amdgpu_kernel void @invert_true_phi_cond_break_loop(i32 %arg) #0 {
bb:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %tmp = sub i32 %id, %arg
  br label %bb1

bb1:                                              ; preds = %Flow, %bb
  %lsr.iv = phi i32 [ undef, %bb ], [ %tmp2, %Flow ]
  %lsr.iv.next = add i32 %lsr.iv, 1
  %cmp0 = icmp slt i32 %lsr.iv.next, 0
  br i1 %cmp0, label %bb4, label %Flow

bb4:                                              ; preds = %bb1
  %load = load volatile i32, i32 addrspace(1)* undef, align 4
  %cmp1 = icmp sge i32 %tmp, %load
  br label %Flow

Flow:                                             ; preds = %bb4, %bb1
  %tmp2 = phi i32 [ %lsr.iv.next, %bb4 ], [ undef, %bb1 ]
  %tmp3 = phi i1 [ %cmp1, %bb4 ], [ true, %bb1 ]
  br i1 %tmp3, label %bb1, label %bb9

bb9:                                              ; preds = %Flow
  store volatile i32 7, i32 addrspace(3)* undef
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
