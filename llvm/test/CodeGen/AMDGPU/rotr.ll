; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck --check-prefix=R600 -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}rotr_i32:
; R600: BIT_ALIGN_INT

; SI: v_alignbit_b32
define amdgpu_kernel void @rotr_i32(ptr addrspace(1) %in, i32 %x, i32 %y) {
entry:
  %tmp0 = sub i32 32, %y
  %tmp1 = shl i32 %x, %tmp0
  %tmp2 = lshr i32 %x, %y
  %tmp3 = or i32 %tmp1, %tmp2
  store i32 %tmp3, ptr addrspace(1) %in
  ret void
}

; FUNC-LABEL: {{^}}rotr_v2i32:
; R600: BIT_ALIGN_INT
; R600: BIT_ALIGN_INT

; SI: v_alignbit_b32
; SI: v_alignbit_b32
define amdgpu_kernel void @rotr_v2i32(ptr addrspace(1) %in, <2 x i32> %x, <2 x i32> %y) {
entry:
  %tmp0 = sub <2 x i32> <i32 32, i32 32>, %y
  %tmp1 = shl <2 x i32> %x, %tmp0
  %tmp2 = lshr <2 x i32> %x, %y
  %tmp3 = or <2 x i32> %tmp1, %tmp2
  store <2 x i32> %tmp3, ptr addrspace(1) %in
  ret void
}

; FUNC-LABEL: {{^}}rotr_v4i32:
; R600: BIT_ALIGN_INT
; R600: BIT_ALIGN_INT
; R600: BIT_ALIGN_INT
; R600: BIT_ALIGN_INT

; SI: v_alignbit_b32
; SI: v_alignbit_b32
; SI: v_alignbit_b32
; SI: v_alignbit_b32
define amdgpu_kernel void @rotr_v4i32(ptr addrspace(1) %in, <4 x i32> %x, <4 x i32> %y) {
entry:
  %tmp0 = sub <4 x i32> <i32 32, i32 32, i32 32, i32 32>, %y
  %tmp1 = shl <4 x i32> %x, %tmp0
  %tmp2 = lshr <4 x i32> %x, %y
  %tmp3 = or <4 x i32> %tmp1, %tmp2
  store <4 x i32> %tmp3, ptr addrspace(1) %in
  ret void
}

; GCN-LABEL: @test_rotr_i16
; GCN: global_load_ushort [[X:v[0-9]+]]
; GCN: global_load_ushort [[D:v[0-9]+]]
; GCN: v_sub_nc_u16_e64 [[NX:v[0-9]+]], 0, [[X]]
; GCN: v_and_b32_e32 [[XAND:v[0-9]+]], 15, [[X]]
; GCN: v_and_b32_e32 [[NXAND:v[0-9]+]], 15, [[NX]]
; GCN: v_lshrrev_b16_e64 [[LO:v[0-9]+]], [[XAND]], [[D]]
; GCN: v_lshlrev_b16_e64 [[HI:v[0-9]+]], [[NXAND]], [[D]]
; GCN: v_or_b32_e32 [[RES:v[0-9]+]], [[LO]], [[HI]]
; GCN: global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RES]]

declare i16 @llvm.fshr.i16(i16, i16, i16)

define void @test_rotr_i16(ptr addrspace(1) nocapture readonly %sourceA, ptr addrspace(1) nocapture readonly %sourceB, ptr addrspace(1) nocapture %destValues) {
entry:
  %arrayidx = getelementptr inbounds i16, ptr addrspace(1) %sourceA, i64 16
  %a = load i16, ptr addrspace(1) %arrayidx
  %arrayidx2 = getelementptr inbounds i16, ptr addrspace(1) %sourceB, i64 24
  %b = load i16, ptr addrspace(1) %arrayidx2
  %c = tail call i16 @llvm.fshr.i16(i16 %a, i16 %a, i16 %b)
  %arrayidx5 = getelementptr inbounds i16, ptr addrspace(1) %destValues, i64 4
  store i16 %c, ptr addrspace(1) %arrayidx5
  ret void
}
