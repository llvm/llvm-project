; RUN: llc -march=amdgcn -mcpu=gfx1030 -verify-machineinstrs -asm-verbose=0 < %s | FileCheck --check-prefixes=GCN,GFX10,GFX10-ASM %s
; RUN: llc -march=amdgcn -mcpu=gfx1030 -verify-machineinstrs < %s -filetype=obj | llvm-objdump -d --arch-name=amdgcn --mcpu=gfx1030 --symbolize-operands - | FileCheck --check-prefixes=GCN,GFX10,GFX10-DIS %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck --check-prefix=GFX8 %s

; GFX8-NOT: s_inst_prefetch
; GFX8-NOT: .palign 6

; GCN-LABEL: test_loop_64
; GFX10:          s_movk_i32 s{{[0-9]+}}, 0x400
; GFX10-DIS-NEXT: {{^$}}
; GFX10-ASM-NEXT: [[L1:.LBB[0-9_]+]]:
; GFX10-DIS-NEXT: <[[L1:L[0-9]+]]>:
; GFX10:          s_sleep 0
; GFX10:          s_cbranch_scc0 [[L1]]
; GFX10-NEXT:     s_endpgm
define amdgpu_kernel void @test_loop_64(ptr addrspace(1) nocapture %arg) {
bb:
  br label %bb2

bb1:                                              ; preds = %bb2
  ret void

bb2:                                              ; preds = %bb2, %bb
  %tmp1 = phi i32 [ 0, %bb ], [ %tmp2, %bb2 ]
  %tmp2 = add nuw nsw i32 %tmp1, 1
  %tmp3 = icmp eq i32 %tmp2, 1024
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  br i1 %tmp3, label %bb1, label %bb2
}

; GCN-LABEL: test_loop_128
; GFX10:          s_movk_i32 s{{[0-9]+}}, 0x400
; GFX10-ASM-NEXT: .p2align 6
; GFX10-DIS-NEXT: s_nop 0
; GFX10-NOT:      s_inst_prefetch
; GFX10-ASM:      [[L1:.LBB[0-9_]+]]:
; GFX10-DIS:      <[[L1:L[0-9]+]]>:
; GFX10:          s_sleep 0
; GFX10:          s_cbranch_scc0 [[L1]]
; GFX10-NEXT:     s_endpgm
define amdgpu_kernel void @test_loop_128(ptr addrspace(1) nocapture %arg) {
bb:
  br label %bb2

bb1:                                              ; preds = %bb2
  ret void

bb2:                                              ; preds = %bb2, %bb
  %tmp1 = phi i32 [ 0, %bb ], [ %tmp2, %bb2 ]
  %tmp2 = add nuw nsw i32 %tmp1, 1
  %tmp3 = icmp eq i32 %tmp2, 1024
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  br i1 %tmp3, label %bb1, label %bb2
}

; GCN-LABEL: test_loop_192
; GFX10:          s_movk_i32 s{{[0-9]+}}, 0x400
; GFX10-NEXT:     s_inst_prefetch 0x1
; GFX10-ASM-NEXT: .p2align 6
; GFX10-DIS-NEXT: s_nop 0
; GFX10-NOT:      s_inst_prefetch
; GFX10-ASM:      [[L1:.LBB[0-9_]+]]:
; GFX10-DIS:      <[[L1:L[0-9]+]]>:
; GFX10:          s_sleep 0
; GFX10:          s_cbranch_scc0 [[L1]]
; GFX10-NEXT:     s_inst_prefetch 0x2
; GFX10-NEXT:     s_endpgm
define amdgpu_kernel void @test_loop_192(ptr addrspace(1) nocapture %arg) {
bb:
  br label %bb2

bb1:                                              ; preds = %bb2
  ret void

bb2:                                              ; preds = %bb2, %bb
  %tmp1 = phi i32 [ 0, %bb ], [ %tmp2, %bb2 ]
  %tmp2 = add nuw nsw i32 %tmp1, 1
  %tmp3 = icmp eq i32 %tmp2, 1024
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  br i1 %tmp3, label %bb1, label %bb2
}

; GCN-LABEL: test_loop_256
; GFX10:          s_movk_i32 s{{[0-9]+}}, 0x400
; GFX10-DIS-NEXT: {{^$}}
; GFX10-ASM-NEXT: [[L1:.LBB[0-9_]+]]:
; GFX10-DIS-NEXT: <[[L1:L[0-9]+]]>:
; GFX10:          s_sleep 0
; GFX10:          s_cbranch_scc0 [[L1]]
; GFX10-NEXT:     s_endpgm
define amdgpu_kernel void @test_loop_256(ptr addrspace(1) nocapture %arg) {
bb:
  br label %bb2

bb1:                                              ; preds = %bb2
  ret void

bb2:                                              ; preds = %bb2, %bb
  %tmp1 = phi i32 [ 0, %bb ], [ %tmp2, %bb2 ]
  %tmp2 = add nuw nsw i32 %tmp1, 1
  %tmp3 = icmp eq i32 %tmp2, 1024
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  br i1 %tmp3, label %bb1, label %bb2
}

; GCN-LABEL: test_loop_prefetch_inner_outer
; GFX10:          s_inst_prefetch 0x1
; GFX10-ASM-NEXT: .p2align 6
; GFX10-DIS-NEXT: s_nop 0
; GFX10-NOT:      s_inst_prefetch
; GFX10-ASM:      [[L1:.LBB[0-9_]+]]:
; GFX10-DIS:      <[[L1:L[0-9]+]]>:
; GFX10-NOT:      s_inst_prefetch
; GFX10-ASM:      .p2align 6
; GFX10-DIS:      s_nop 0
; GFX10-NOT:      s_inst_prefetch
; GFX10-ASM:      [[L2:.LBB[0-9_]+]]:
; GFX10-DIS:      <[[L2:L[0-9]+]]>:
; GFX10-NOT:      s_inst_prefetch
; GFX10:          s_sleep 0
; GFX10:          s_cbranch_scc{{[01]}} [[L2]]
; GFX10-NOT:      s_inst_prefetch
; GFX10:          s_cbranch_scc{{[01]}} [[L1]]
; GFX10-NEXT:     s_inst_prefetch 0x2
; GFX10-NEXT:     s_endpgm
define amdgpu_kernel void @test_loop_prefetch_inner_outer(ptr addrspace(1) nocapture %arg) {
bb:
  br label %bb2

bb1:
  ret void

bb2:
  %tmp1 = phi i32 [ 0, %bb ], [ %tmp2, %bb4 ]
  %tmp2 = add nuw nsw i32 %tmp1, 1
  %tmp3 = icmp eq i32 %tmp2, 1024
  br label %bb3

bb3:
  %tmp4 = phi i32 [ 0, %bb2 ], [ %tmp5, %bb3 ]
  %tmp5 = add nuw nsw i32 %tmp4, 1
  %tmp6 = icmp eq i32 %tmp5, 1024
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  br i1 %tmp6, label %bb4, label %bb3

bb4:
  br i1 %tmp3, label %bb1, label %bb2
}

; GCN-LABEL: test_loop_prefetch_inner_outer_noouter
; GFX10-NOT:      .p2align 6
; GFX10-NOT:      s_nop
; GFX10-NOT:      s_inst_prefetch
; GFX10-ASM:      [[L0:.LBB[0-9_]+]]:
; GFX10-DIS:      <[[L0:L[0-9]+]]>:
; GFX10:          s_inst_prefetch 0x1
; GFX10-ASM-NEXT: .p2align 6
; GFX10-DIS-NEXT: s_nop 0
; GFX10-NOT:      s_inst_prefetch
; GFX10-ASM:      [[L1:.LBB[0-9_]+]]:
; GFX10-DIS:      <[[L1:L[0-9]+]]>:
; GFX10-NOT:      s_inst_prefetch
; GFX10-ASM:      .p2align 6
; GFX10-DIS:      s_nop 0
; GFX10-NOT:      s_inst_prefetch
; GFX10-ASM:      [[L2:.LBB[0-9_]+]]:
; GFX10-DIS:      <[[L2:L[0-9]+]]>:
; GFX10-NOT:      s_inst_prefetch
; GFX10:          s_sleep 0
; GFX10:          s_cbranch_scc{{[01]}} [[L2]]
; GFX10-NOT:      s_inst_prefetch
; GFX10:          s_cbranch_scc{{[01]}} [[L1]]
; GFX10-NEXT:     s_inst_prefetch 0x2
; GFX10:          s_cbranch_scc{{[01]}} [[L0]]
; GFX10-NEXT:     s_endpgm
define amdgpu_kernel void @test_loop_prefetch_inner_outer_noouter(ptr addrspace(1) nocapture %arg) {
bb:
  br label %bb2

bb1:
  ret void

bb2:
  %tmp1 = phi i32 [ 0, %bb ], [ %tmp2, %bb6 ]
  %tmp2 = add nuw nsw i32 %tmp1, 1
  %tmp3 = icmp eq i32 %tmp2, 1024
  br label %bb3

bb3:
  %tmp4 = phi i32 [ 0, %bb2 ], [ %tmp5, %bb5 ]
  %tmp5 = add nuw nsw i32 %tmp4, 1
  %tmp6 = icmp eq i32 %tmp5, 1024
  br label %bb4

bb4:
  %tmp7 = phi i32 [ 0, %bb3 ], [ %tmp8, %bb4 ]
  %tmp8 = add nuw nsw i32 %tmp7, 1
  %tmp9 = icmp eq i32 %tmp8, 1024
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  br i1 %tmp9, label %bb5, label %bb4

bb5:
  br i1 %tmp6, label %bb6, label %bb3

bb6:
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  tail call void @llvm.amdgcn.s.sleep(i32 0)
  br i1 %tmp3, label %bb1, label %bb2
}

declare void @llvm.amdgcn.s.sleep(i32)
