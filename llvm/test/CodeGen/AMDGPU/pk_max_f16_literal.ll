; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX9 %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX10 %s
; RUN: llc -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX10 %s

; GCN-LABEL: {{^}}test_pk_max_f16_literal_0_1:
; GCN: v_pk_max_f16 v{{[0-9]+}}, v{{[0-9]+}}, 1.0 op_sel:[0,1] op_sel_hi:[1,0]{{$}}
define amdgpu_kernel void @test_pk_max_f16_literal_0_1(ptr addrspace(1) nocapture %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds <2 x half>, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load <2 x half>, ptr addrspace(1) %tmp2, align 4
  %tmp4 = tail call <2 x half> @llvm.maxnum.v2f16(<2 x half> %tmp3, <2 x half> <half 0xH0000, half 0xH3C00>)
  store <2 x half> %tmp4, ptr addrspace(1) %tmp2, align 4
  ret void
}

; GCN-LABEL: {{^}}test_pk_max_f16_literal_1_0:
; GCN: v_pk_max_f16 v{{[0-9]+}}, v{{[0-9]+}}, 1.0{{$}}
define amdgpu_kernel void @test_pk_max_f16_literal_1_0(ptr addrspace(1) nocapture %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds <2 x half>, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load <2 x half>, ptr addrspace(1) %tmp2, align 4
  %tmp4 = tail call <2 x half> @llvm.maxnum.v2f16(<2 x half> %tmp3, <2 x half> <half 0xH3C00, half 0xH0000>)
  store <2 x half> %tmp4, ptr addrspace(1) %tmp2, align 4
  ret void
}

; GCN-LABEL: {{^}}test_pk_max_f16_literal_1_1:
; GCN: v_pk_max_f16 v{{[0-9]+}}, v{{[0-9]+}}, 1.0 op_sel_hi:[1,0]{{$}}
define amdgpu_kernel void @test_pk_max_f16_literal_1_1(ptr addrspace(1) nocapture %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds <2 x half>, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load <2 x half>, ptr addrspace(1) %tmp2, align 4
  %tmp4 = tail call <2 x half> @llvm.maxnum.v2f16(<2 x half> %tmp3, <2 x half> <half 0xH3C00, half 0xH3C00>)
  store <2 x half> %tmp4, ptr addrspace(1) %tmp2, align 4
  ret void
}

; GCN-LABEL: {{^}}test_pk_max_f16_literal_0_m1:
; GCN: v_pk_max_f16 v{{[0-9]+}}, v{{[0-9]+}}, -1.0 op_sel:[0,1] op_sel_hi:[1,0]{{$}}
define amdgpu_kernel void @test_pk_max_f16_literal_0_m1(ptr addrspace(1) nocapture %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds <2 x half>, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load <2 x half>, ptr addrspace(1) %tmp2, align 4
  %tmp4 = tail call <2 x half> @llvm.maxnum.v2f16(<2 x half> %tmp3, <2 x half> <half 0xH0000, half 0xHBC00>)
  store <2 x half> %tmp4, ptr addrspace(1) %tmp2, align 4
  ret void
}

; GCN-LABEL: {{^}}test_pk_max_f16_literal_m1_0:
; GCN: v_pk_max_f16 v{{[0-9]+}}, v{{[0-9]+}}, -1.0{{$}}
define amdgpu_kernel void @test_pk_max_f16_literal_m1_0(ptr addrspace(1) nocapture %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds <2 x half>, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load <2 x half>, ptr addrspace(1) %tmp2, align 4
  %tmp4 = tail call <2 x half> @llvm.maxnum.v2f16(<2 x half> %tmp3, <2 x half> <half 0xHBC00, half 0xH0000>)
  store <2 x half> %tmp4, ptr addrspace(1) %tmp2, align 4
  ret void
}

; GCN-LABEL: {{^}}test_pk_max_f16_literal_m1_m1:
; GCN: v_pk_max_f16 v{{[0-9]+}}, v{{[0-9]+}}, -1.0 op_sel_hi:[1,0]{{$}}
define amdgpu_kernel void @test_pk_max_f16_literal_m1_m1(ptr addrspace(1) nocapture %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds <2 x half>, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load <2 x half>, ptr addrspace(1) %tmp2, align 4
  %tmp4 = tail call <2 x half> @llvm.maxnum.v2f16(<2 x half> %tmp3, <2 x half> <half 0xHBC00, half 0xHBC00>)
  store <2 x half> %tmp4, ptr addrspace(1) %tmp2, align 4
  ret void
}

; GCN-LABEL: {{^}}test_pk_max_f16_literal_0_0:
; GCN: v_pk_max_f16 v{{[0-9]+}}, v{{[0-9]+}}, 0{{$}}
define amdgpu_kernel void @test_pk_max_f16_literal_0_0(ptr addrspace(1) nocapture %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds <2 x half>, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load <2 x half>, ptr addrspace(1) %tmp2, align 4
  %tmp4 = tail call <2 x half> @llvm.maxnum.v2f16(<2 x half> %tmp3, <2 x half> <half 0xH0000, half 0xH0000>)
  store <2 x half> %tmp4, ptr addrspace(1) %tmp2, align 4
  ret void
}

; GCN-LABEL: {{^}}test_pk_max_f16_literal_0_41c8:
; GFX9:  s_mov_b32 [[C:s[0-9]+]], 0x41c80000
; GFX9:  v_pk_max_f16 v{{[0-9]+}}, v{{[0-9]+}}, [[C]]{{$}}
; GFX10: v_pk_max_f16 v{{[0-9]+}}, 0x41c80000, v{{[0-9]+}}{{$}}
define amdgpu_kernel void @test_pk_max_f16_literal_0_41c8(ptr addrspace(1) nocapture %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds <2 x half>, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load <2 x half>, ptr addrspace(1) %tmp2, align 4
  %tmp4 = tail call <2 x half> @llvm.maxnum.v2f16(<2 x half> %tmp3, <2 x half> <half 0xH0000, half 0xH41C8>)
  store <2 x half> %tmp4, ptr addrspace(1) %tmp2, align 4
  ret void
}

; GCN-LABEL: {{^}}test_pk_max_f16_literal_41c8_0:
; GFX9:  s_movk_i32 [[C:s[0-9]+]], 0x41c8
; GFX9:  v_pk_max_f16 v{{[0-9]+}}, v{{[0-9]+}}, [[C]]{{$}}
; GFX10: v_pk_max_f16 v{{[0-9]+}}, 0x41c8, v{{[0-9]+}}{{$}}
define amdgpu_kernel void @test_pk_max_f16_literal_41c8_0(ptr addrspace(1) nocapture %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds <2 x half>, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load <2 x half>, ptr addrspace(1) %tmp2, align 4
  %tmp4 = tail call <2 x half> @llvm.maxnum.v2f16(<2 x half> %tmp3, <2 x half> <half 0xH41C8, half 0xH0>)
  store <2 x half> %tmp4, ptr addrspace(1) %tmp2, align 4
  ret void
}

; GCN-LABEL: {{^}}test_pk_max_f16_literal_42ca_41c8:
; GFX9:  s_mov_b32 [[C:s[0-9]+]], 0x41c842ca
; GFX9:  v_pk_max_f16 v{{[0-9]+}}, v{{[0-9]+}}, [[C]]{{$}}
; GFX10: v_pk_max_f16 v{{[0-9]+}}, 0x41c842ca, v{{[0-9]+}}{{$}}
define amdgpu_kernel void @test_pk_max_f16_literal_42ca_41c8(ptr addrspace(1) nocapture %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds <2 x half>, ptr addrspace(1) %arg, i64 %tmp1
  %tmp3 = load <2 x half>, ptr addrspace(1) %tmp2, align 4
  %tmp4 = tail call <2 x half> @llvm.maxnum.v2f16(<2 x half> %tmp3, <2 x half> <half 0xH42CA, half 0xH41C8>)
  store <2 x half> %tmp4, ptr addrspace(1) %tmp2, align 4
  ret void
}

declare <2 x half> @llvm.maxnum.v2f16(<2 x half>, <2 x half>)
declare i32 @llvm.amdgcn.workitem.id.x()
