; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -O3 < %s | FileCheck -check-prefix=GCN %s
; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds < %s | FileCheck %s

@a = internal unnamed_addr addrspace(3) global [64 x i32] poison, align 4
@b = internal unnamed_addr addrspace(3) global [64 x i32] poison, align 4
@c = internal unnamed_addr addrspace(3) global [64 x i32] poison, align 4

; FIXME: Should combine the DS instructions into ds_write2 and ds_read2. This
; does not happen because when SILoadStoreOptimizer is run, the reads and writes
; are not adjacent. They are only moved later by MachineScheduler.

define amdgpu_kernel void @no_clobber_ds_load_stores_x2(ptr addrspace(1) %arg, i32 %i) {
; CHECK-LABEL: define amdgpu_kernel void @no_clobber_ds_load_stores_x2(
; CHECK-SAME: ptr addrspace(1) [[ARG:%.*]], i32 [[I:%.*]]) #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:  bb:
; CHECK-NEXT:    store i32 1, ptr addrspace(3) @llvm.amdgcn.kernel.no_clobber_ds_load_stores_x2.lds, align 16, !alias.scope !2, !noalias !5
; CHECK-NEXT:    [[GEP_A:%.*]] = getelementptr inbounds [64 x i32], ptr addrspace(3) @llvm.amdgcn.kernel.no_clobber_ds_load_stores_x2.lds, i32 0, i32 [[I]]
; CHECK-NEXT:    [[VAL_A:%.*]] = load i32, ptr addrspace(3) [[GEP_A]], align 4, !alias.scope !2, !noalias !5
; CHECK-NEXT:    store i32 2, ptr addrspace(3) getelementptr inbounds ([[LLVM_AMDGCN_KERNEL_NO_CLOBBER_DS_LOAD_STORES_X2_LDS_T:%.*]], ptr addrspace(3) @llvm.amdgcn.kernel.no_clobber_ds_load_stores_x2.lds, i32 0, i32 1), align 16, !alias.scope !5, !noalias !2
; CHECK-NEXT:    [[GEP_B:%.*]] = getelementptr inbounds [64 x i32], ptr addrspace(3) getelementptr inbounds ([[LLVM_AMDGCN_KERNEL_NO_CLOBBER_DS_LOAD_STORES_X2_LDS_T]], ptr addrspace(3) @llvm.amdgcn.kernel.no_clobber_ds_load_stores_x2.lds, i32 0, i32 1), i32 0, i32 [[I]]
; CHECK-NEXT:    [[VAL_B:%.*]] = load i32, ptr addrspace(3) [[GEP_B]], align 4, !alias.scope !5, !noalias !2
; CHECK-NEXT:    [[VAL:%.*]] = add i32 [[VAL_A]], [[VAL_B]]
; CHECK-NEXT:    store i32 [[VAL]], ptr addrspace(1) [[ARG]], align 4
; CHECK-NEXT:    ret void
;
; GCN-LABEL: no_clobber_ds_load_stores_x2:
; GCN:       ; %bb.0: ; %bb
; GCN-NEXT:    s_load_dword s0, s[4:5], 0x2c
; GCN-NEXT:    v_mov_b32_e32 v0, 1
; GCN-NEXT:    v_mov_b32_e32 v1, 0
; GCN-NEXT:    v_mov_b32_e32 v2, 2
; GCN-NEXT:    ds_write_b32 v1, v0
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    s_lshl_b32 s0, s0, 2
; GCN-NEXT:    v_mov_b32_e32 v0, s0
; GCN-NEXT:    ds_write_b32 v1, v2 offset:256
; GCN-NEXT:    ds_read_b32 v2, v0
; GCN-NEXT:    ds_read_b32 v0, v0 offset:256
; GCN-NEXT:    s_load_dwordx2 s[0:1], s[4:5], 0x24
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    v_add_u32_e32 v0, v2, v0
; GCN-NEXT:    global_store_dword v1, v0, s[0:1]
; GCN-NEXT:    s_endpgm
bb:
  store i32 1, ptr addrspace(3) @a, align 4
  %gep.a = getelementptr inbounds [64 x i32], ptr addrspace(3) @a, i32 0, i32 %i
  %val.a = load i32, ptr addrspace(3) %gep.a, align 4
  store i32 2, ptr addrspace(3) @b, align 4
  %gep.b = getelementptr inbounds [64 x i32], ptr addrspace(3) @b, i32 0, i32 %i
  %val.b = load i32, ptr addrspace(3) %gep.b, align 4
  %val = add i32 %val.a, %val.b
  store i32 %val, ptr addrspace(1) %arg, align 4
  ret void
}

define amdgpu_kernel void @no_clobber_ds_load_stores_x3(ptr addrspace(1) %arg, i32 %i) {
; CHECK-LABEL: define amdgpu_kernel void @no_clobber_ds_load_stores_x3(
; CHECK-SAME: ptr addrspace(1) [[ARG:%.*]], i32 [[I:%.*]]) #[[ATTR1:[0-9]+]] {
; CHECK-NEXT:  bb:
; CHECK-NEXT:    store i32 1, ptr addrspace(3) @llvm.amdgcn.kernel.no_clobber_ds_load_stores_x3.lds, align 16, !alias.scope !7, !noalias !10
; CHECK-NEXT:    [[GEP_A:%.*]] = getelementptr inbounds [64 x i32], ptr addrspace(3) @llvm.amdgcn.kernel.no_clobber_ds_load_stores_x3.lds, i32 0, i32 [[I]]
; CHECK-NEXT:    [[VAL_A:%.*]] = load i32, ptr addrspace(3) [[GEP_A]], align 4, !alias.scope !7, !noalias !10
; CHECK-NEXT:    store i32 2, ptr addrspace(3) getelementptr inbounds ([[LLVM_AMDGCN_KERNEL_NO_CLOBBER_DS_LOAD_STORES_X3_LDS_T:%.*]], ptr addrspace(3) @llvm.amdgcn.kernel.no_clobber_ds_load_stores_x3.lds, i32 0, i32 1), align 16, !alias.scope !13, !noalias !14
; CHECK-NEXT:    [[GEP_B:%.*]] = getelementptr inbounds [64 x i32], ptr addrspace(3) getelementptr inbounds ([[LLVM_AMDGCN_KERNEL_NO_CLOBBER_DS_LOAD_STORES_X3_LDS_T]], ptr addrspace(3) @llvm.amdgcn.kernel.no_clobber_ds_load_stores_x3.lds, i32 0, i32 1), i32 0, i32 [[I]]
; CHECK-NEXT:    [[VAL_B:%.*]] = load i32, ptr addrspace(3) [[GEP_B]], align 4, !alias.scope !13, !noalias !14
; CHECK-NEXT:    store i32 3, ptr addrspace(3) getelementptr inbounds ([[LLVM_AMDGCN_KERNEL_NO_CLOBBER_DS_LOAD_STORES_X3_LDS_T]], ptr addrspace(3) @llvm.amdgcn.kernel.no_clobber_ds_load_stores_x3.lds, i32 0, i32 2), align 16, !alias.scope !15, !noalias !16
; CHECK-NEXT:    [[GEP_C:%.*]] = getelementptr inbounds [64 x i32], ptr addrspace(3) getelementptr inbounds ([[LLVM_AMDGCN_KERNEL_NO_CLOBBER_DS_LOAD_STORES_X3_LDS_T]], ptr addrspace(3) @llvm.amdgcn.kernel.no_clobber_ds_load_stores_x3.lds, i32 0, i32 2), i32 0, i32 [[I]]
; CHECK-NEXT:    [[VAL_C:%.*]] = load i32, ptr addrspace(3) [[GEP_C]], align 4, !alias.scope !15, !noalias !16
; CHECK-NEXT:    [[VAL_1:%.*]] = add i32 [[VAL_A]], [[VAL_B]]
; CHECK-NEXT:    [[VAL:%.*]] = add i32 [[VAL_1]], [[VAL_C]]
; CHECK-NEXT:    store i32 [[VAL]], ptr addrspace(1) [[ARG]], align 4
; CHECK-NEXT:    ret void
;
; GCN-LABEL: no_clobber_ds_load_stores_x3:
; GCN:       ; %bb.0: ; %bb
; GCN-NEXT:    s_load_dword s0, s[4:5], 0x2c
; GCN-NEXT:    v_mov_b32_e32 v1, 0
; GCN-NEXT:    v_mov_b32_e32 v2, 2
; GCN-NEXT:    v_mov_b32_e32 v0, 1
; GCN-NEXT:    ds_write_b32 v1, v2 offset:256
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    s_lshl_b32 s0, s0, 2
; GCN-NEXT:    v_mov_b32_e32 v2, 3
; GCN-NEXT:    ds_write_b32 v1, v0
; GCN-NEXT:    v_mov_b32_e32 v0, s0
; GCN-NEXT:    ds_write_b32 v1, v2 offset:512
; GCN-NEXT:    ds_read_b32 v2, v0
; GCN-NEXT:    ds_read_b32 v3, v0 offset:256
; GCN-NEXT:    ds_read_b32 v0, v0 offset:512
; GCN-NEXT:    s_load_dwordx2 s[0:1], s[4:5], 0x24
; GCN-NEXT:    s_waitcnt lgkmcnt(0)
; GCN-NEXT:    v_add_u32_e32 v2, v2, v3
; GCN-NEXT:    v_add_u32_e32 v0, v2, v0
; GCN-NEXT:    global_store_dword v1, v0, s[0:1]
; GCN-NEXT:    s_endpgm
bb:
  store i32 1, ptr addrspace(3) @a, align 4
  %gep.a = getelementptr inbounds [64 x i32], ptr addrspace(3) @a, i32 0, i32 %i
  %val.a = load i32, ptr addrspace(3) %gep.a, align 4
  store i32 2, ptr addrspace(3) @b, align 4
  %gep.b = getelementptr inbounds [64 x i32], ptr addrspace(3) @b, i32 0, i32 %i
  %val.b = load i32, ptr addrspace(3) %gep.b, align 4
  store i32 3, ptr addrspace(3) @c, align 4
  %gep.c = getelementptr inbounds [64 x i32], ptr addrspace(3) @c, i32 0, i32 %i
  %val.c = load i32, ptr addrspace(3) %gep.c, align 4
  %val.1 = add i32 %val.a, %val.b
  %val = add i32 %val.1, %val.c
  store i32 %val, ptr addrspace(1) %arg, align 4
  ret void
}

; CHECK: !0 = !{i32 0, i32 1}
; CHECK: !1 = !{i32 1, !"amdgpu.lowered_lds", i32 1}
; CHECK: !2 = !{!3}
; CHECK: !3 = distinct !{!3, !4}
; CHECK: !4 = distinct !{!4}
; CHECK: !5 = !{!6}
; CHECK: !6 = distinct !{!6, !4}
; CHECK: !7 = !{!8}
; CHECK: !8 = distinct !{!8, !9}
; CHECK: !9 = distinct !{!9}
; CHECK: !10 = !{!11, !12}
; CHECK: !11 = distinct !{!11, !9}
; CHECK: !12 = distinct !{!12, !9}
; CHECK: !13 = !{!11}
; CHECK: !14 = !{!8, !12}
; CHECK: !15 = !{!12}
; CHECK: !16 = !{!8, !11}
