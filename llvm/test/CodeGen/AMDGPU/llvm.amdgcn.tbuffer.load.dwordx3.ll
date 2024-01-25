; RUN: llc -global-isel=0 < %s -mtriple=amdgcn -mcpu=gfx600 -verify-machineinstrs | FileCheck %s -check-prefixes=GCN,SI
; RUN: llc -global-isel=0 < %s -mtriple=amdgcn -mcpu=gfx700 -verify-machineinstrs | FileCheck %s -check-prefixes=GCN,GCNX3

; GCN-LABEL: {{^}}tbuffer_raw_load_immoffs_x3:
; SI: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 format:[BUF_DATA_FORMAT_32_32_32_32,BUF_NUM_FORMAT_UINT] offset:42
; GCNX3: tbuffer_load_format_xyz {{v\[[0-9]+:[0-9]+\]}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 format:[BUF_DATA_FORMAT_32_32_32_32,BUF_NUM_FORMAT_UINT] offset:42
define amdgpu_vs <3 x float> @tbuffer_raw_load_immoffs_x3(<4 x i32> inreg) {
main_body:
    %vdata   = call <3 x i32> @llvm.amdgcn.raw.tbuffer.load.v3i32(<4 x i32> %0, i32 42, i32 0, i32 78, i32 0)
    %vdata.f = bitcast <3 x i32> %vdata to <3 x float>
    ret <3 x float> %vdata.f
}


; GCN-LABEL: {{^}}tbuffer_struct_load_immoffs_x3:
; GCN: v_mov_b32_e32 [[ZEROREG:v[0-9]+]], 0
; SI: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, [[ZEROREG]], {{s\[[0-9]+:[0-9]+\]}}, 0 format:[BUF_DATA_FORMAT_32_32_32_32,BUF_NUM_FORMAT_UINT] idxen offset:42
; GCNX3: tbuffer_load_format_xyz {{v\[[0-9]+:[0-9]+\]}}, [[ZEROREG]], {{s\[[0-9]+:[0-9]+\]}}, 0 format:[BUF_DATA_FORMAT_32_32_32_32,BUF_NUM_FORMAT_UINT] idxen offset:42
define amdgpu_vs <3 x float> @tbuffer_struct_load_immoffs_x3(<4 x i32> inreg) {
main_body:
    %vdata   = call <3 x i32> @llvm.amdgcn.struct.tbuffer.load.v3i32(<4 x i32> %0, i32 0, i32 42, i32 0, i32 78, i32 0)
    %vdata.f = bitcast <3 x i32> %vdata to <3 x float>
    ret <3 x float> %vdata.f
}


; GCN-LABEL: {{^}}tbuffer_load_format_immoffs_x3:
; SI: tbuffer_load_format_xyzw {{v\[[0-9]+:[0-9]+\]}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 format:[BUF_DATA_FORMAT_32_32_32_32,BUF_NUM_FORMAT_UINT] offset:42
; GCNX3: tbuffer_load_format_xyz {{v\[[0-9]+:[0-9]+\]}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 format:[BUF_DATA_FORMAT_32_32_32_32,BUF_NUM_FORMAT_UINT] offset:42
define amdgpu_vs <3 x float> @tbuffer_load_format_immoffs_x3(<4 x i32> inreg) {
main_body:
    %vdata   = call <3 x i32> @llvm.amdgcn.tbuffer.load.v3i32(<4 x i32> %0, i32 0, i32 0, i32 0, i32 42, i32 14, i32 4, i1 0, i1 0)
    %vdata.f = bitcast <3 x i32> %vdata to <3 x float>
    ret <3 x float> %vdata.f
}

declare <3 x i32> @llvm.amdgcn.raw.tbuffer.load.v3i32(<4 x i32>, i32, i32, i32, i32)
declare <3 x i32> @llvm.amdgcn.struct.tbuffer.load.v3i32(<4 x i32>, i32, i32, i32, i32, i32)
declare <3 x i32> @llvm.amdgcn.tbuffer.load.v3i32(<4 x i32>, i32, i32, i32, i32, i32, i32, i1, i1)

