; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX9 %s

; GCN-LABEL: {{^}}test_pk_max_f16_literal_0_1:
; GFX9: v_pk_max_f16 v{{[0-9]+}}, v{{[0-9]+}}, 1.0 op_sel:[0,1] op_sel_hi:[1,0]{{$}}
define amdgpu_kernel void @test_pk_max_f16_literal_0_1(<2 x half> addrspace(1)* nocapture %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds <2 x half>, <2 x half> addrspace(1)* %arg, i64 %tmp1
  %tmp3 = load <2 x half>, <2 x half> addrspace(1)* %tmp2, align 4
  %tmp4 = tail call <2 x half> @llvm.maxnum.v2f16(<2 x half> %tmp3, <2 x half> <half 0xH0000, half 0xH3C00>)
  store <2 x half> %tmp4, <2 x half> addrspace(1)* %tmp2, align 4
  ret void
}

; GCN-LABEL: {{^}}test_pk_max_f16_literal_1_0:
; GFX9: v_pk_max_f16 v{{[0-9]+}}, v{{[0-9]+}}, 1.0{{$}}
define amdgpu_kernel void @test_pk_max_f16_literal_1_0(<2 x half> addrspace(1)* nocapture %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds <2 x half>, <2 x half> addrspace(1)* %arg, i64 %tmp1
  %tmp3 = load <2 x half>, <2 x half> addrspace(1)* %tmp2, align 4
  %tmp4 = tail call <2 x half> @llvm.maxnum.v2f16(<2 x half> %tmp3, <2 x half> <half 0xH3C00, half 0xH0000>) 
  store <2 x half> %tmp4, <2 x half> addrspace(1)* %tmp2, align 4
  ret void
}

; GCN-LABEL: {{^}}test_pk_max_f16_literal_1_1:
; GFX9: v_pk_max_f16 v{{[0-9]+}}, v{{[0-9]+}}, 1.0 op_sel_hi:[1,0]{{$}}
define amdgpu_kernel void @test_pk_max_f16_literal_1_1(<2 x half> addrspace(1)* nocapture %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds <2 x half>, <2 x half> addrspace(1)* %arg, i64 %tmp1
  %tmp3 = load <2 x half>, <2 x half> addrspace(1)* %tmp2, align 4
  %tmp4 = tail call <2 x half> @llvm.maxnum.v2f16(<2 x half> %tmp3, <2 x half> <half 0xH3C00, half 0xH3C00>)
  store <2 x half> %tmp4, <2 x half> addrspace(1)* %tmp2, align 4
  ret void
}

; GCN-LABEL: {{^}}test_pk_max_f16_literal_0_m1:
; GFX9: v_pk_max_f16 v{{[0-9]+}}, -1.0, v{{[0-9]+}} op_sel:[1,0] op_sel_hi:[0,1]{{$}}
define amdgpu_kernel void @test_pk_max_f16_literal_0_m1(<2 x half> addrspace(1)* nocapture %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds <2 x half>, <2 x half> addrspace(1)* %arg, i64 %tmp1
  %tmp3 = load <2 x half>, <2 x half> addrspace(1)* %tmp2, align 4
  %tmp4 = tail call <2 x half> @llvm.maxnum.v2f16(<2 x half> %tmp3, <2 x half> <half 0xH0000, half 0xHBC00>)
  store <2 x half> %tmp4, <2 x half> addrspace(1)* %tmp2, align 4
  ret void
}

; GCN-LABEL: {{^}}test_pk_max_f16_literal_m1_0:
; GFX9: v_pk_max_f16 v{{[0-9]+}}, v{{[0-9]+}}, -1.0{{$}}
define amdgpu_kernel void @test_pk_max_f16_literal_m1_0(<2 x half> addrspace(1)* nocapture %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds <2 x half>, <2 x half> addrspace(1)* %arg, i64 %tmp1
  %tmp3 = load <2 x half>, <2 x half> addrspace(1)* %tmp2, align 4
  %tmp4 = tail call <2 x half> @llvm.maxnum.v2f16(<2 x half> %tmp3, <2 x half> <half 0xHBC00, half 0xH0000>)
  store <2 x half> %tmp4, <2 x half> addrspace(1)* %tmp2, align 4
  ret void
}

; GCN-LABEL: {{^}}test_pk_max_f16_literal_m1_m1:
; GFX9: v_pk_max_f16 v{{[0-9]+}}, v{{[0-9]+}}, -1.0 op_sel_hi:[1,0]{{$}}
define amdgpu_kernel void @test_pk_max_f16_literal_m1_m1(<2 x half> addrspace(1)* nocapture %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds <2 x half>, <2 x half> addrspace(1)* %arg, i64 %tmp1
  %tmp3 = load <2 x half>, <2 x half> addrspace(1)* %tmp2, align 4
  %tmp4 = tail call <2 x half> @llvm.maxnum.v2f16(<2 x half> %tmp3, <2 x half> <half 0xHBC00, half 0xHBC00>)
  store <2 x half> %tmp4, <2 x half> addrspace(1)* %tmp2, align 4
  ret void
}

; GCN-LABEL: {{^}}test_pk_max_f16_literal_0_0:
; GFX9: v_pk_max_f16 v{{[0-9]+}}, v{{[0-9]+}}, 0{{$}}
define amdgpu_kernel void @test_pk_max_f16_literal_0_0(<2 x half> addrspace(1)* nocapture %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tmp1 = zext i32 %tmp to i64
  %tmp2 = getelementptr inbounds <2 x half>, <2 x half> addrspace(1)* %arg, i64 %tmp1
  %tmp3 = load <2 x half>, <2 x half> addrspace(1)* %tmp2, align 4
  %tmp4 = tail call <2 x half> @llvm.maxnum.v2f16(<2 x half> %tmp3, <2 x half> <half 0xH0000, half 0xH0000>)
  store <2 x half> %tmp4, <2 x half> addrspace(1)* %tmp2, align 4
  ret void
}

declare <2 x half> @llvm.maxnum.v2f16(<2 x half>, <2 x half>)
declare i32 @llvm.amdgcn.workitem.id.x()
