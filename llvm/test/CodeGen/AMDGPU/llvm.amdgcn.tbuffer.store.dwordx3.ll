;RUN: llc -global-isel=0 < %s -march=amdgcn -mcpu=gfx700 -verify-machineinstrs | FileCheck %s -check-prefixes=GCN

; GCN-LABEL: {{^}}tbuffer_raw_store_immoffs_x3:
; GCN: tbuffer_store_format_xyz v[0:2], off, s[0:3], 0 format:[BUF_DATA_FORMAT_16_16,BUF_NUM_FORMAT_FLOAT] offset:42
define amdgpu_ps void @tbuffer_raw_store_immoffs_x3(<4 x i32> inreg, <3 x float>) {
main_body:
  %in1 = bitcast <3 x float> %1 to <3 x i32>
  call void @llvm.amdgcn.raw.tbuffer.store.v3i32(<3 x i32> %in1, <4 x i32> %0, i32 42, i32 0, i32 117, i32 0)
  ret void
}


; GCN-LABEL: {{^}}tbuffer_struct_store_immoffs_x3:
; GCN: v_mov_b32_e32 [[ZEROREG:v[0-9]+]], 0
; GCN: tbuffer_store_format_xyz v[0:2], [[ZEROREG]], s[0:3], 0 format:[BUF_DATA_FORMAT_16_16,BUF_NUM_FORMAT_FLOAT] idxen offset:42
define amdgpu_ps void @tbuffer_struct_store_immoffs_x3(<4 x i32> inreg, <3 x float>) {
main_body:
  %in1 = bitcast <3 x float> %1 to <3 x i32>
  call void @llvm.amdgcn.struct.tbuffer.store.v3i32(<3 x i32> %in1, <4 x i32> %0, i32 0, i32 42, i32 0, i32 117, i32 0)
  ret void
}

; GCN-LABEL: {{^}}tbuffer_store_immoffs_x3:
; GCN: tbuffer_store_format_xyz v[0:2], off, s[0:3], 0 format:[BUF_DATA_FORMAT_16_16,BUF_NUM_FORMAT_FLOAT] offset:42
define amdgpu_ps void @tbuffer_store_immoffs_x3(<4 x i32> inreg, <3 x float>) {
main_body:
  %in1 = bitcast <3 x float> %1 to <3 x i32>
  call void @llvm.amdgcn.tbuffer.store.v3i32(<3 x i32> %in1, <4 x i32> %0, i32 0, i32 0, i32 0, i32 42, i32 5, i32 7, i1 0, i1 0)
  ret void
}

declare void @llvm.amdgcn.raw.tbuffer.store.v3i32(<3 x i32>, <4 x i32>, i32, i32, i32, i32) #0
declare void @llvm.amdgcn.struct.tbuffer.store.v3i32(<3 x i32>, <4 x i32>, i32, i32, i32, i32, i32) #0
declare void @llvm.amdgcn.tbuffer.store.v3i32(<3 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i1, i1) #0

