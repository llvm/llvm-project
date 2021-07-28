; RUN: opt -mtriple=amdgcn-- -S -structurizecfg -si-annotate-control-flow %s | FileCheck -check-prefix=OPT %s
; RUN: llc -march=amdgcn -verify-machineinstrs -disable-block-placement < %s | FileCheck -check-prefix=GCN %s

; Uses llvm.amdgcn.break

define amdgpu_kernel void @break_loop(i32 %arg) #0 {
; OPT-LABEL: @break_loop(
; OPT-NEXT:  bb:
; OPT-NEXT:    [[ID:%.*]] = call i32 @llvm.amdgcn.workitem.id.x()
; OPT-NEXT:    [[MY_TMP:%.*]] = sub i32 [[ID]], [[ARG:%.*]]
; OPT-NEXT:    br label [[BB1:%.*]]
; OPT:       bb1:
; OPT-NEXT:    [[PHI_BROKEN:%.*]] = phi i64 [ [[TMP2:%.*]], [[FLOW:%.*]] ], [ 0, [[BB:%.*]] ]
; OPT-NEXT:    [[LSR_IV:%.*]] = phi i32 [ undef, [[BB]] ], [ [[LSR_IV_NEXT:%.*]], [[FLOW]] ]
; OPT-NEXT:    [[LSR_IV_NEXT]] = add i32 [[LSR_IV]], 1
; OPT-NEXT:    [[CMP0:%.*]] = icmp slt i32 [[LSR_IV_NEXT]], 0
; OPT-NEXT:    br i1 [[CMP0]], label [[BB4:%.*]], label [[FLOW]]
; OPT:       bb4:
; OPT-NEXT:    [[LOAD:%.*]] = load volatile i32, i32 addrspace(1)* undef, align 4
; OPT-NEXT:    [[CMP1:%.*]] = icmp slt i32 [[MY_TMP]], [[LOAD]]
; OPT-NEXT:    [[TMP0:%.*]] = xor i1 [[CMP1]], true
; OPT-NEXT:    br label [[FLOW]]
; OPT:       Flow:
; OPT-NEXT:    [[TMP1:%.*]] = phi i1 [ [[TMP0]], [[BB4]] ], [ true, [[BB1]] ]
; OPT-NEXT:    [[TMP2]] = call i64 @llvm.amdgcn.if.break.i64(i1 [[TMP1]], i64 [[PHI_BROKEN]])
; OPT-NEXT:    [[TMP3:%.*]] = call i1 @llvm.amdgcn.loop.i64(i64 [[TMP2]])
; OPT-NEXT:    br i1 [[TMP3]], label [[BB9:%.*]], label [[BB1]]
; OPT:       bb9:
; OPT-NEXT:    call void @llvm.amdgcn.end.cf.i64(i64 [[TMP2]])
; OPT-NEXT:    ret void
;
; GCN-LABEL: break_loop:
; GCN:       ; %bb.0: ; %bb
; GCN-NEXT:    s_load_dword s3, s[0:1], 0x9
; GCN-NEXT:    s_mov_b64 s[0:1], 0
; GCN-NEXT:    s_mov_b32 s2, -1
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    v_subrev_i32_e32 v0, vcc, s3, v0
; GCN-NEXT:    s_mov_b32 s3, 0xf000
; GCN-NEXT:    ; implicit-def: $sgpr4_sgpr5
; GCN-NEXT:    ; implicit-def: $sgpr6
; GCN-NEXT:  BB0_1: ; %bb1
; GCN-NEXT:    ; =>This Inner Loop Header: Depth=1
; GCN-NEXT:    s_add_i32 s6, s6, 1
; GCN-NEXT:    s_or_b64 s[4:5], s[4:5], exec
; GCN-NEXT:    s_cmp_gt_i32 s6, -1
; GCN-NEXT:    s_cbranch_scc1 BB0_3
; GCN-NEXT:  ; %bb.2: ; %bb4
; GCN-NEXT:    ; in Loop: Header=BB0_1 Depth=1
; GCN-NEXT:    buffer_load_dword v1, off, s[0:3], 0 glc
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    v_cmp_ge_i32_e32 vcc, v0, v1
; GCN-NEXT:    s_andn2_b64 s[4:5], s[4:5], exec
; GCN-NEXT:    s_and_b64 s[8:9], vcc, exec
; GCN-NEXT:    s_or_b64 s[4:5], s[4:5], s[8:9]
; GCN-NEXT:  BB0_3: ; %Flow
; GCN-NEXT:    ; in Loop: Header=BB0_1 Depth=1
; GCN-NEXT:    s_and_b64 s[8:9], exec, s[4:5]
; GCN-NEXT:    s_or_b64 s[0:1], s[8:9], s[0:1]
; GCN-NEXT:    s_andn2_b64 exec, exec, s[0:1]
; GCN-NEXT:    s_cbranch_execnz BB0_1
; GCN-NEXT:  ; %bb.4: ; %bb9
; GCN-NEXT:    s_endpgm
bb:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %my.tmp = sub i32 %id, %arg
  br label %bb1

bb1:
  %lsr.iv = phi i32 [ undef, %bb ], [ %lsr.iv.next, %bb4 ]
  %lsr.iv.next = add i32 %lsr.iv, 1
  %cmp0 = icmp slt i32 %lsr.iv.next, 0
  br i1 %cmp0, label %bb4, label %bb9

bb4:
  %load = load volatile i32, i32 addrspace(1)* undef, align 4
  %cmp1 = icmp slt i32 %my.tmp, %load
  br i1 %cmp1, label %bb1, label %bb9

bb9:
  ret void
}

define amdgpu_kernel void @undef_phi_cond_break_loop(i32 %arg) #0 {
; OPT-LABEL: @undef_phi_cond_break_loop(
; OPT-NEXT:  bb:
; OPT-NEXT:    [[ID:%.*]] = call i32 @llvm.amdgcn.workitem.id.x()
; OPT-NEXT:    [[MY_TMP:%.*]] = sub i32 [[ID]], [[ARG:%.*]]
; OPT-NEXT:    br label [[BB1:%.*]]
; OPT:       bb1:
; OPT-NEXT:    [[PHI_BROKEN:%.*]] = phi i64 [ [[TMP0:%.*]], [[FLOW:%.*]] ], [ 0, [[BB:%.*]] ]
; OPT-NEXT:    [[LSR_IV:%.*]] = phi i32 [ undef, [[BB]] ], [ [[LSR_IV_NEXT:%.*]], [[FLOW]] ]
; OPT-NEXT:    [[LSR_IV_NEXT]] = add i32 [[LSR_IV]], 1
; OPT-NEXT:    [[CMP0:%.*]] = icmp slt i32 [[LSR_IV_NEXT]], 0
; OPT-NEXT:    br i1 [[CMP0]], label [[BB4:%.*]], label [[FLOW]]
; OPT:       bb4:
; OPT-NEXT:    [[LOAD:%.*]] = load volatile i32, i32 addrspace(1)* undef, align 4
; OPT-NEXT:    [[CMP1:%.*]] = icmp sge i32 [[MY_TMP]], [[LOAD]]
; OPT-NEXT:    br label [[FLOW]]
; OPT:       Flow:
; OPT-NEXT:    [[MY_TMP3:%.*]] = phi i1 [ [[CMP1]], [[BB4]] ], [ undef, [[BB1]] ]
; OPT-NEXT:    [[TMP0]] = call i64 @llvm.amdgcn.if.break.i64(i1 [[MY_TMP3]], i64 [[PHI_BROKEN]])
; OPT-NEXT:    [[TMP1:%.*]] = call i1 @llvm.amdgcn.loop.i64(i64 [[TMP0]])
; OPT-NEXT:    br i1 [[TMP1]], label [[BB9:%.*]], label [[BB1]]
; OPT:       bb9:
; OPT-NEXT:    call void @llvm.amdgcn.end.cf.i64(i64 [[TMP0]])
; OPT-NEXT:    store volatile i32 7, i32 addrspace(3)* undef
; OPT-NEXT:    ret void
;
; GCN-LABEL: undef_phi_cond_break_loop:
; GCN:       ; %bb.0: ; %bb
; GCN-NEXT:    s_load_dword s3, s[0:1], 0x9
; GCN-NEXT:    s_mov_b64 s[0:1], 0
; GCN-NEXT:    s_mov_b32 s2, -1
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    v_subrev_i32_e32 v0, vcc, s3, v0
; GCN-NEXT:    s_mov_b32 s3, 0xf000
; GCN-NEXT:    ; implicit-def: $sgpr4_sgpr5
; GCN-NEXT:    ; implicit-def: $sgpr6
; GCN-NEXT:  BB1_1: ; %bb1
; GCN-NEXT:    ; =>This Inner Loop Header: Depth=1
; GCN-NEXT:    s_andn2_b64 s[4:5], s[4:5], exec
; GCN-NEXT:    s_cmp_gt_i32 s6, -1
; GCN-NEXT:    s_cbranch_scc1 BB1_3
; GCN-NEXT:  ; %bb.2: ; %bb4
; GCN-NEXT:    ; in Loop: Header=BB1_1 Depth=1
; GCN-NEXT:    buffer_load_dword v1, off, s[0:3], 0 glc
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    v_cmp_ge_i32_e32 vcc, v0, v1
; GCN-NEXT:    s_andn2_b64 s[4:5], s[4:5], exec
; GCN-NEXT:    s_and_b64 s[8:9], vcc, exec
; GCN-NEXT:    s_or_b64 s[4:5], s[4:5], s[8:9]
; GCN-NEXT:  BB1_3: ; %Flow
; GCN-NEXT:    ; in Loop: Header=BB1_1 Depth=1
; GCN-NEXT:    s_add_i32 s6, s6, 1
; GCN-NEXT:    s_and_b64 s[8:9], exec, s[4:5]
; GCN-NEXT:    s_or_b64 s[0:1], s[8:9], s[0:1]
; GCN-NEXT:    s_andn2_b64 exec, exec, s[0:1]
; GCN-NEXT:    s_cbranch_execnz BB1_1
; GCN-NEXT:  ; %bb.4: ; %bb9
; GCN-NEXT:    s_or_b64 exec, exec, s[0:1]
; GCN-NEXT:    v_mov_b32_e32 v0, 7
; GCN-NEXT:    s_mov_b32 m0, -1
; GCN-NEXT:    ds_write_b32 v0, v0
; GCN-NEXT:    s_endpgm
bb:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %my.tmp = sub i32 %id, %arg
  br label %bb1

bb1:                                              ; preds = %Flow, %bb
  %lsr.iv = phi i32 [ undef, %bb ], [ %my.tmp2, %Flow ]
  %lsr.iv.next = add i32 %lsr.iv, 1
  %cmp0 = icmp slt i32 %lsr.iv.next, 0
  br i1 %cmp0, label %bb4, label %Flow

bb4:                                              ; preds = %bb1
  %load = load volatile i32, i32 addrspace(1)* undef, align 4
  %cmp1 = icmp sge i32 %my.tmp, %load
  br label %Flow

Flow:                                             ; preds = %bb4, %bb1
  %my.tmp2 = phi i32 [ %lsr.iv.next, %bb4 ], [ undef, %bb1 ]
  %my.tmp3 = phi i1 [ %cmp1, %bb4 ], [ undef, %bb1 ]
  br i1 %my.tmp3, label %bb9, label %bb1

bb9:                                              ; preds = %Flow
  store volatile i32 7, i32 addrspace(3)* undef
  ret void
}

; FIXME: ConstantExpr compare of address to null folds away
@lds = addrspace(3) global i32 undef

define amdgpu_kernel void @constexpr_phi_cond_break_loop(i32 %arg) #0 {
; OPT-LABEL: @constexpr_phi_cond_break_loop(
; OPT-NEXT:  bb:
; OPT-NEXT:    [[ID:%.*]] = call i32 @llvm.amdgcn.workitem.id.x()
; OPT-NEXT:    [[MY_TMP:%.*]] = sub i32 [[ID]], [[ARG:%.*]]
; OPT-NEXT:    br label [[BB1:%.*]]
; OPT:       bb1:
; OPT-NEXT:    [[PHI_BROKEN:%.*]] = phi i64 [ [[TMP0:%.*]], [[FLOW:%.*]] ], [ 0, [[BB:%.*]] ]
; OPT-NEXT:    [[LSR_IV:%.*]] = phi i32 [ undef, [[BB]] ], [ [[LSR_IV_NEXT:%.*]], [[FLOW]] ]
; OPT-NEXT:    [[LSR_IV_NEXT]] = add i32 [[LSR_IV]], 1
; OPT-NEXT:    [[CMP0:%.*]] = icmp slt i32 [[LSR_IV_NEXT]], 0
; OPT-NEXT:    br i1 [[CMP0]], label [[BB4:%.*]], label [[FLOW]]
; OPT:       bb4:
; OPT-NEXT:    [[LOAD:%.*]] = load volatile i32, i32 addrspace(1)* undef, align 4
; OPT-NEXT:    [[CMP1:%.*]] = icmp sge i32 [[MY_TMP]], [[LOAD]]
; OPT-NEXT:    br label [[FLOW]]
; OPT:       Flow:
; OPT-NEXT:    [[MY_TMP3:%.*]] = phi i1 [ [[CMP1]], [[BB4]] ], [ icmp ne (i32 addrspace(3)* inttoptr (i32 4 to i32 addrspace(3)*), i32 addrspace(3)* @lds), [[BB1]] ]
; OPT-NEXT:    [[TMP0]] = call i64 @llvm.amdgcn.if.break.i64(i1 [[MY_TMP3]], i64 [[PHI_BROKEN]])
; OPT-NEXT:    [[TMP1:%.*]] = call i1 @llvm.amdgcn.loop.i64(i64 [[TMP0]])
; OPT-NEXT:    br i1 [[TMP1]], label [[BB9:%.*]], label [[BB1]]
; OPT:       bb9:
; OPT-NEXT:    call void @llvm.amdgcn.end.cf.i64(i64 [[TMP0]])
; OPT-NEXT:    store volatile i32 7, i32 addrspace(3)* undef
; OPT-NEXT:    ret void
;
; GCN-LABEL: constexpr_phi_cond_break_loop:
; GCN:       ; %bb.0: ; %bb
; GCN-NEXT:    s_load_dword s3, s[0:1], 0x9
; GCN-NEXT:    s_mov_b64 s[0:1], 0
; GCN-NEXT:    s_mov_b32 s2, -1
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    v_subrev_i32_e32 v0, vcc, s3, v0
; GCN-NEXT:    s_mov_b32 s3, 0xf000
; GCN-NEXT:    ; implicit-def: $sgpr4_sgpr5
; GCN-NEXT:    ; implicit-def: $sgpr6
; GCN-NEXT:  BB2_1: ; %bb1
; GCN-NEXT:    ; =>This Inner Loop Header: Depth=1
; GCN-NEXT:    s_or_b64 s[4:5], s[4:5], exec
; GCN-NEXT:    s_cmp_gt_i32 s6, -1
; GCN-NEXT:    s_cbranch_scc1 BB2_3
; GCN-NEXT:  ; %bb.2: ; %bb4
; GCN-NEXT:    ; in Loop: Header=BB2_1 Depth=1
; GCN-NEXT:    buffer_load_dword v1, off, s[0:3], 0 glc
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    v_cmp_ge_i32_e32 vcc, v0, v1
; GCN-NEXT:    s_andn2_b64 s[4:5], s[4:5], exec
; GCN-NEXT:    s_and_b64 s[8:9], vcc, exec
; GCN-NEXT:    s_or_b64 s[4:5], s[4:5], s[8:9]
; GCN-NEXT:  BB2_3: ; %Flow
; GCN-NEXT:    ; in Loop: Header=BB2_1 Depth=1
; GCN-NEXT:    s_add_i32 s6, s6, 1
; GCN-NEXT:    s_and_b64 s[8:9], exec, s[4:5]
; GCN-NEXT:    s_or_b64 s[0:1], s[8:9], s[0:1]
; GCN-NEXT:    s_andn2_b64 exec, exec, s[0:1]
; GCN-NEXT:    s_cbranch_execnz BB2_1
; GCN-NEXT:  ; %bb.4: ; %bb9
; GCN-NEXT:    s_or_b64 exec, exec, s[0:1]
; GCN-NEXT:    v_mov_b32_e32 v0, 7
; GCN-NEXT:    s_mov_b32 m0, -1
; GCN-NEXT:    ds_write_b32 v0, v0
; GCN-NEXT:    s_endpgm
bb:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %my.tmp = sub i32 %id, %arg
  br label %bb1

bb1:                                              ; preds = %Flow, %bb
  %lsr.iv = phi i32 [ undef, %bb ], [ %my.tmp2, %Flow ]
  %lsr.iv.next = add i32 %lsr.iv, 1
  %cmp0 = icmp slt i32 %lsr.iv.next, 0
  br i1 %cmp0, label %bb4, label %Flow

bb4:                                              ; preds = %bb1
  %load = load volatile i32, i32 addrspace(1)* undef, align 4
  %cmp1 = icmp sge i32 %my.tmp, %load
  br label %Flow

Flow:                                             ; preds = %bb4, %bb1
  %my.tmp2 = phi i32 [ %lsr.iv.next, %bb4 ], [ undef, %bb1 ]
  %my.tmp3 = phi i1 [ %cmp1, %bb4 ], [ icmp ne (i32 addrspace(3)* inttoptr (i32 4 to i32 addrspace(3)*), i32 addrspace(3)* @lds), %bb1 ]
  br i1 %my.tmp3, label %bb9, label %bb1

bb9:                                              ; preds = %Flow
  store volatile i32 7, i32 addrspace(3)* undef
  ret void
}

define amdgpu_kernel void @true_phi_cond_break_loop(i32 %arg) #0 {
; OPT-LABEL: @true_phi_cond_break_loop(
; OPT-NEXT:  bb:
; OPT-NEXT:    [[ID:%.*]] = call i32 @llvm.amdgcn.workitem.id.x()
; OPT-NEXT:    [[MY_TMP:%.*]] = sub i32 [[ID]], [[ARG:%.*]]
; OPT-NEXT:    br label [[BB1:%.*]]
; OPT:       bb1:
; OPT-NEXT:    [[PHI_BROKEN:%.*]] = phi i64 [ [[TMP0:%.*]], [[FLOW:%.*]] ], [ 0, [[BB:%.*]] ]
; OPT-NEXT:    [[LSR_IV:%.*]] = phi i32 [ undef, [[BB]] ], [ [[LSR_IV_NEXT:%.*]], [[FLOW]] ]
; OPT-NEXT:    [[LSR_IV_NEXT]] = add i32 [[LSR_IV]], 1
; OPT-NEXT:    [[CMP0:%.*]] = icmp slt i32 [[LSR_IV_NEXT]], 0
; OPT-NEXT:    br i1 [[CMP0]], label [[BB4:%.*]], label [[FLOW]]
; OPT:       bb4:
; OPT-NEXT:    [[LOAD:%.*]] = load volatile i32, i32 addrspace(1)* undef, align 4
; OPT-NEXT:    [[CMP1:%.*]] = icmp sge i32 [[MY_TMP]], [[LOAD]]
; OPT-NEXT:    br label [[FLOW]]
; OPT:       Flow:
; OPT-NEXT:    [[MY_TMP3:%.*]] = phi i1 [ [[CMP1]], [[BB4]] ], [ true, [[BB1]] ]
; OPT-NEXT:    [[TMP0]] = call i64 @llvm.amdgcn.if.break.i64(i1 [[MY_TMP3]], i64 [[PHI_BROKEN]])
; OPT-NEXT:    [[TMP1:%.*]] = call i1 @llvm.amdgcn.loop.i64(i64 [[TMP0]])
; OPT-NEXT:    br i1 [[TMP1]], label [[BB9:%.*]], label [[BB1]]
; OPT:       bb9:
; OPT-NEXT:    call void @llvm.amdgcn.end.cf.i64(i64 [[TMP0]])
; OPT-NEXT:    store volatile i32 7, i32 addrspace(3)* undef
; OPT-NEXT:    ret void
;
; GCN-LABEL: true_phi_cond_break_loop:
; GCN:       ; %bb.0: ; %bb
; GCN-NEXT:    s_load_dword s3, s[0:1], 0x9
; GCN-NEXT:    s_mov_b64 s[0:1], 0
; GCN-NEXT:    s_mov_b32 s2, -1
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    v_subrev_i32_e32 v0, vcc, s3, v0
; GCN-NEXT:    s_mov_b32 s3, 0xf000
; GCN-NEXT:    ; implicit-def: $sgpr4_sgpr5
; GCN-NEXT:    ; implicit-def: $sgpr6
; GCN-NEXT:  BB3_1: ; %bb1
; GCN-NEXT:    ; =>This Inner Loop Header: Depth=1
; GCN-NEXT:    s_or_b64 s[4:5], s[4:5], exec
; GCN-NEXT:    s_cmp_gt_i32 s6, -1
; GCN-NEXT:    s_cbranch_scc1 BB3_3
; GCN-NEXT:  ; %bb.2: ; %bb4
; GCN-NEXT:    ; in Loop: Header=BB3_1 Depth=1
; GCN-NEXT:    buffer_load_dword v1, off, s[0:3], 0 glc
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    v_cmp_ge_i32_e32 vcc, v0, v1
; GCN-NEXT:    s_andn2_b64 s[4:5], s[4:5], exec
; GCN-NEXT:    s_and_b64 s[8:9], vcc, exec
; GCN-NEXT:    s_or_b64 s[4:5], s[4:5], s[8:9]
; GCN-NEXT:  BB3_3: ; %Flow
; GCN-NEXT:    ; in Loop: Header=BB3_1 Depth=1
; GCN-NEXT:    s_add_i32 s6, s6, 1
; GCN-NEXT:    s_and_b64 s[8:9], exec, s[4:5]
; GCN-NEXT:    s_or_b64 s[0:1], s[8:9], s[0:1]
; GCN-NEXT:    s_andn2_b64 exec, exec, s[0:1]
; GCN-NEXT:    s_cbranch_execnz BB3_1
; GCN-NEXT:  ; %bb.4: ; %bb9
; GCN-NEXT:    s_or_b64 exec, exec, s[0:1]
; GCN-NEXT:    v_mov_b32_e32 v0, 7
; GCN-NEXT:    s_mov_b32 m0, -1
; GCN-NEXT:    ds_write_b32 v0, v0
; GCN-NEXT:    s_endpgm
bb:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %my.tmp = sub i32 %id, %arg
  br label %bb1

bb1:                                              ; preds = %Flow, %bb
  %lsr.iv = phi i32 [ undef, %bb ], [ %my.tmp2, %Flow ]
  %lsr.iv.next = add i32 %lsr.iv, 1
  %cmp0 = icmp slt i32 %lsr.iv.next, 0
  br i1 %cmp0, label %bb4, label %Flow

bb4:                                              ; preds = %bb1
  %load = load volatile i32, i32 addrspace(1)* undef, align 4
  %cmp1 = icmp sge i32 %my.tmp, %load
  br label %Flow

Flow:                                             ; preds = %bb4, %bb1
  %my.tmp2 = phi i32 [ %lsr.iv.next, %bb4 ], [ undef, %bb1 ]
  %my.tmp3 = phi i1 [ %cmp1, %bb4 ], [ true, %bb1 ]
  br i1 %my.tmp3, label %bb9, label %bb1

bb9:                                              ; preds = %Flow
  store volatile i32 7, i32 addrspace(3)* undef
  ret void
}

define amdgpu_kernel void @false_phi_cond_break_loop(i32 %arg) #0 {
; OPT-LABEL: @false_phi_cond_break_loop(
; OPT-NEXT:  bb:
; OPT-NEXT:    [[ID:%.*]] = call i32 @llvm.amdgcn.workitem.id.x()
; OPT-NEXT:    [[MY_TMP:%.*]] = sub i32 [[ID]], [[ARG:%.*]]
; OPT-NEXT:    br label [[BB1:%.*]]
; OPT:       bb1:
; OPT-NEXT:    [[PHI_BROKEN:%.*]] = phi i64 [ [[TMP0:%.*]], [[FLOW:%.*]] ], [ 0, [[BB:%.*]] ]
; OPT-NEXT:    [[LSR_IV:%.*]] = phi i32 [ undef, [[BB]] ], [ [[LSR_IV_NEXT:%.*]], [[FLOW]] ]
; OPT-NEXT:    [[LSR_IV_NEXT]] = add i32 [[LSR_IV]], 1
; OPT-NEXT:    [[CMP0:%.*]] = icmp slt i32 [[LSR_IV_NEXT]], 0
; OPT-NEXT:    br i1 [[CMP0]], label [[BB4:%.*]], label [[FLOW]]
; OPT:       bb4:
; OPT-NEXT:    [[LOAD:%.*]] = load volatile i32, i32 addrspace(1)* undef, align 4
; OPT-NEXT:    [[CMP1:%.*]] = icmp sge i32 [[MY_TMP]], [[LOAD]]
; OPT-NEXT:    br label [[FLOW]]
; OPT:       Flow:
; OPT-NEXT:    [[MY_TMP3:%.*]] = phi i1 [ [[CMP1]], [[BB4]] ], [ false, [[BB1]] ]
; OPT-NEXT:    [[TMP0]] = call i64 @llvm.amdgcn.if.break.i64(i1 [[MY_TMP3]], i64 [[PHI_BROKEN]])
; OPT-NEXT:    [[TMP1:%.*]] = call i1 @llvm.amdgcn.loop.i64(i64 [[TMP0]])
; OPT-NEXT:    br i1 [[TMP1]], label [[BB9:%.*]], label [[BB1]]
; OPT:       bb9:
; OPT-NEXT:    call void @llvm.amdgcn.end.cf.i64(i64 [[TMP0]])
; OPT-NEXT:    store volatile i32 7, i32 addrspace(3)* undef
; OPT-NEXT:    ret void
;
; GCN-LABEL: false_phi_cond_break_loop:
; GCN:       ; %bb.0: ; %bb
; GCN-NEXT:    s_load_dword s3, s[0:1], 0x9
; GCN-NEXT:    s_mov_b64 s[0:1], 0
; GCN-NEXT:    s_mov_b32 s2, -1
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    v_subrev_i32_e32 v0, vcc, s3, v0
; GCN-NEXT:    s_mov_b32 s3, 0xf000
; GCN-NEXT:    ; implicit-def: $sgpr4_sgpr5
; GCN-NEXT:    ; implicit-def: $sgpr6
; GCN-NEXT:  BB4_1: ; %bb1
; GCN-NEXT:    ; =>This Inner Loop Header: Depth=1
; GCN-NEXT:    s_andn2_b64 s[4:5], s[4:5], exec
; GCN-NEXT:    s_cmp_gt_i32 s6, -1
; GCN-NEXT:    s_cbranch_scc1 BB4_3
; GCN-NEXT:  ; %bb.2: ; %bb4
; GCN-NEXT:    ; in Loop: Header=BB4_1 Depth=1
; GCN-NEXT:    buffer_load_dword v1, off, s[0:3], 0 glc
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    v_cmp_ge_i32_e32 vcc, v0, v1
; GCN-NEXT:    s_andn2_b64 s[4:5], s[4:5], exec
; GCN-NEXT:    s_and_b64 s[8:9], vcc, exec
; GCN-NEXT:    s_or_b64 s[4:5], s[4:5], s[8:9]
; GCN-NEXT:  BB4_3: ; %Flow
; GCN-NEXT:    ; in Loop: Header=BB4_1 Depth=1
; GCN-NEXT:    s_add_i32 s6, s6, 1
; GCN-NEXT:    s_and_b64 s[8:9], exec, s[4:5]
; GCN-NEXT:    s_or_b64 s[0:1], s[8:9], s[0:1]
; GCN-NEXT:    s_andn2_b64 exec, exec, s[0:1]
; GCN-NEXT:    s_cbranch_execnz BB4_1
; GCN-NEXT:  ; %bb.4: ; %bb9
; GCN-NEXT:    s_or_b64 exec, exec, s[0:1]
; GCN-NEXT:    v_mov_b32_e32 v0, 7
; GCN-NEXT:    s_mov_b32 m0, -1
; GCN-NEXT:    ds_write_b32 v0, v0
; GCN-NEXT:    s_endpgm
bb:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %my.tmp = sub i32 %id, %arg
  br label %bb1

bb1:                                              ; preds = %Flow, %bb
  %lsr.iv = phi i32 [ undef, %bb ], [ %my.tmp2, %Flow ]
  %lsr.iv.next = add i32 %lsr.iv, 1
  %cmp0 = icmp slt i32 %lsr.iv.next, 0
  br i1 %cmp0, label %bb4, label %Flow

bb4:                                              ; preds = %bb1
  %load = load volatile i32, i32 addrspace(1)* undef, align 4
  %cmp1 = icmp sge i32 %my.tmp, %load
  br label %Flow

Flow:                                             ; preds = %bb4, %bb1
  %my.tmp2 = phi i32 [ %lsr.iv.next, %bb4 ], [ undef, %bb1 ]
  %my.tmp3 = phi i1 [ %cmp1, %bb4 ], [ false, %bb1 ]
  br i1 %my.tmp3, label %bb9, label %bb1

bb9:                                              ; preds = %Flow
  store volatile i32 7, i32 addrspace(3)* undef
  ret void
}

; Swap order of branches in flow block so that the true phi is
; continue.

define amdgpu_kernel void @invert_true_phi_cond_break_loop(i32 %arg) #0 {
; OPT-LABEL: @invert_true_phi_cond_break_loop(
; OPT-NEXT:  bb:
; OPT-NEXT:    [[ID:%.*]] = call i32 @llvm.amdgcn.workitem.id.x()
; OPT-NEXT:    [[MY_TMP:%.*]] = sub i32 [[ID]], [[ARG:%.*]]
; OPT-NEXT:    br label [[BB1:%.*]]
; OPT:       bb1:
; OPT-NEXT:    [[PHI_BROKEN:%.*]] = phi i64 [ [[TMP1:%.*]], [[FLOW:%.*]] ], [ 0, [[BB:%.*]] ]
; OPT-NEXT:    [[LSR_IV:%.*]] = phi i32 [ undef, [[BB]] ], [ [[LSR_IV_NEXT:%.*]], [[FLOW]] ]
; OPT-NEXT:    [[LSR_IV_NEXT]] = add i32 [[LSR_IV]], 1
; OPT-NEXT:    [[CMP0:%.*]] = icmp slt i32 [[LSR_IV_NEXT]], 0
; OPT-NEXT:    br i1 [[CMP0]], label [[BB4:%.*]], label [[FLOW]]
; OPT:       bb4:
; OPT-NEXT:    [[LOAD:%.*]] = load volatile i32, i32 addrspace(1)* undef, align 4
; OPT-NEXT:    [[CMP1:%.*]] = icmp sge i32 [[MY_TMP]], [[LOAD]]
; OPT-NEXT:    br label [[FLOW]]
; OPT:       Flow:
; OPT-NEXT:    [[MY_TMP3:%.*]] = phi i1 [ [[CMP1]], [[BB4]] ], [ true, [[BB1]] ]
; OPT-NEXT:    [[TMP0:%.*]] = xor i1 [[MY_TMP3]], true
; OPT-NEXT:    [[TMP1]] = call i64 @llvm.amdgcn.if.break.i64(i1 [[TMP0]], i64 [[PHI_BROKEN]])
; OPT-NEXT:    [[TMP2:%.*]] = call i1 @llvm.amdgcn.loop.i64(i64 [[TMP1]])
; OPT-NEXT:    br i1 [[TMP2]], label [[BB9:%.*]], label [[BB1]]
; OPT:       bb9:
; OPT-NEXT:    call void @llvm.amdgcn.end.cf.i64(i64 [[TMP1]])
; OPT-NEXT:    store volatile i32 7, i32 addrspace(3)* undef
; OPT-NEXT:    ret void
;
; GCN-LABEL: invert_true_phi_cond_break_loop:
; GCN:       ; %bb.0: ; %bb
; GCN-NEXT:    s_load_dword s3, s[0:1], 0x9
; GCN-NEXT:    s_mov_b64 s[0:1], 0
; GCN-NEXT:    s_mov_b32 s2, -1
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    v_subrev_i32_e32 v0, vcc, s3, v0
; GCN-NEXT:    s_mov_b32 s3, 0xf000
; GCN-NEXT:    ; implicit-def: $sgpr4_sgpr5
; GCN-NEXT:    ; implicit-def: $sgpr6
; GCN-NEXT:  BB5_1: ; %bb1
; GCN-NEXT:    ; =>This Inner Loop Header: Depth=1
; GCN-NEXT:    s_or_b64 s[4:5], s[4:5], exec
; GCN-NEXT:    s_cmp_gt_i32 s6, -1
; GCN-NEXT:    s_cbranch_scc1 BB5_3
; GCN-NEXT:  ; %bb.2: ; %bb4
; GCN-NEXT:    ; in Loop: Header=BB5_1 Depth=1
; GCN-NEXT:    buffer_load_dword v1, off, s[0:3], 0 glc
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    v_cmp_ge_i32_e32 vcc, v0, v1
; GCN-NEXT:    s_andn2_b64 s[4:5], s[4:5], exec
; GCN-NEXT:    s_and_b64 s[8:9], vcc, exec
; GCN-NEXT:    s_or_b64 s[4:5], s[4:5], s[8:9]
; GCN-NEXT:  BB5_3: ; %Flow
; GCN-NEXT:    ; in Loop: Header=BB5_1 Depth=1
; GCN-NEXT:    s_xor_b64 s[8:9], s[4:5], -1
; GCN-NEXT:    s_add_i32 s6, s6, 1
; GCN-NEXT:    s_and_b64 s[8:9], exec, s[8:9]
; GCN-NEXT:    s_or_b64 s[0:1], s[8:9], s[0:1]
; GCN-NEXT:    s_andn2_b64 exec, exec, s[0:1]
; GCN-NEXT:    s_cbranch_execnz BB5_1
; GCN-NEXT:  ; %bb.4: ; %bb9
; GCN-NEXT:    s_or_b64 exec, exec, s[0:1]
; GCN-NEXT:    v_mov_b32_e32 v0, 7
; GCN-NEXT:    s_mov_b32 m0, -1
; GCN-NEXT:    ds_write_b32 v0, v0
; GCN-NEXT:    s_endpgm
bb:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %my.tmp = sub i32 %id, %arg
  br label %bb1

bb1:                                              ; preds = %Flow, %bb
  %lsr.iv = phi i32 [ undef, %bb ], [ %my.tmp2, %Flow ]
  %lsr.iv.next = add i32 %lsr.iv, 1
  %cmp0 = icmp slt i32 %lsr.iv.next, 0
  br i1 %cmp0, label %bb4, label %Flow

bb4:                                              ; preds = %bb1
  %load = load volatile i32, i32 addrspace(1)* undef, align 4
  %cmp1 = icmp sge i32 %my.tmp, %load
  br label %Flow

Flow:                                             ; preds = %bb4, %bb1
  %my.tmp2 = phi i32 [ %lsr.iv.next, %bb4 ], [ undef, %bb1 ]
  %my.tmp3 = phi i1 [ %cmp1, %bb4 ], [ true, %bb1 ]
  br i1 %my.tmp3, label %bb1, label %bb9

bb9:                                              ; preds = %Flow
  store volatile i32 7, i32 addrspace(3)* undef
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
