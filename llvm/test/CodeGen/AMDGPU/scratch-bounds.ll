; RUN: llc -verify-machineinstrs -march=amdgcn -mcpu=gfx900 -mattr=+max-private-element-size-16,+enable-scratch-bounds-checks < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9 %s
; RUN: llc -verify-machineinstrs -march=amdgcn -mcpu=gfx1010 -mattr=+max-private-element-size-16,+enable-scratch-bounds-checks < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX10 %s

; GCN-LABEL: {{^}}bounds_check_load_i32:
; GCN: s_mov_b32 [[BOUNDS:s[0-9]+]], 0x8004
; GFX9: v_cmp_gt_u32_e64 [[BOUNDSMASK:s\[[0-9]+:[0-9]+\]]], [[BOUNDS]], [[OFFSET:v[0-9]+]]
; GFX9: s_and_saveexec_b64 [[EXECMASK:s\[[0-9]+:[0-9]+\]]], [[BOUNDSMASK]]
; GFX10: v_cmp_gt_u32_e64 [[BOUNDSMASK:s[0-9]+]], [[BOUNDS]], [[OFFSET:v[0-9]+]]
; GFX10: s_and_saveexec_b32 [[EXECMASK:s[0-9]+]], [[BOUNDSMASK]]
; GCN: buffer_load_dword [[LOADVALUE:v[0-9]+]], [[OFFSET]], s[{{[0-9]+}}:{{[0-9]+}}], 0 offen{{$}}
; GFX9: s_mov_b64 exec, [[EXECMASK]]
; GFX10: s_mov_b32 exec_lo, [[EXECMASK]]
; GCN: v_cndmask_b32_e64 v{{[0-9]+}}, 0, [[LOADVALUE]], [[BOUNDSMASK]]

define amdgpu_kernel void @bounds_check_load_i32(i32 addrspace(1)* %out, i32 %offset) {
entry:
  %scratch = alloca [8192 x i32], addrspace(5)

  %ptr = getelementptr [8192 x i32], [8192 x i32] addrspace(5)* %scratch, i32 0, i32 %offset
  %value = load i32, i32 addrspace(5)* %ptr
  store i32 %value, i32 addrspace(1)* %out

  ret void
}

; GCN-LABEL: {{^}}bounds_check_store_i32:
; GCN: s_mov_b32 [[BOUNDS:s[0-9]+]], 0x8004
; GFX9: v_cmp_gt_u32_e64 [[BOUNDSMASK:s\[[0-9]+:[0-9]+\]]], [[BOUNDS]], [[OFFSET:v[0-9]+]]
; GFX9: s_and_saveexec_b64 [[EXECMASK:s\[[0-9]+:[0-9]+\]]], [[BOUNDSMASK]]
; GFX10: v_cmp_gt_u32_e64 [[BOUNDSMASK:s[0-9]+]], [[BOUNDS]], [[OFFSET:v[0-9]+]]
; GFX10: s_and_saveexec_b32 [[EXECMASK:s[0-9]+]], [[BOUNDSMASK]]
; GCN: buffer_store_dword v{{[0-9]+}}, [[OFFSET]], s[{{[0-9]+}}:{{[0-9]+}}], 0 offen{{$}}
; GFX9: s_mov_b64 exec, [[EXECMASK]]
; GFX10: s_mov_b32 exec_lo, [[EXECMASK]]

define amdgpu_kernel void @bounds_check_store_i32(i32 addrspace(1)* %out, i32 %value, i32 %offset) {
entry:
  %scratch = alloca [8192 x i32], addrspace(5)

  %ptr = getelementptr [8192 x i32], [8192 x i32] addrspace(5)* %scratch, i32 0, i32 %offset
  store i32 %value, i32 addrspace(5)* %ptr
  store i32 %value, i32 addrspace(1)* %out

  ret void
}

; GCN-LABEL: {{^}}bounds_check_load_i64:
; GCN: s_mov_b32 [[BOUNDS:s[0-9]+]], 0x8008
; GFX9: s_and_saveexec_b64 [[EXECMASK:s\[[0-9]+:[0-9]+\]]], [[BOUNDSMASK:s\[[0-9]+:[0-9]+\]]]
; GFX10: s_and_saveexec_b32 [[EXECMASK:s[0-9]+]], [[BOUNDSMASK:s[0-9]+]]
; GCN: buffer_load_dwordx2 v{{\[}}[[LOADLO:[0-9]+]]:[[LOADHI:[0-9]+]]{{\]}}, [[OFFSET:v[0-9]+]], s[{{[0-9]+}}:{{[0-9]+}}], 0 offen{{$}}
; GFX9: s_mov_b64 exec, [[EXECMASK]]
; GFX10: s_mov_b32 exec_lo, [[EXECMASK]]
; GCN-DAG: v_cndmask_b32_e64 v{{[0-9]+}}, 0, v[[LOADLO]], [[BOUNDSMASK]]
; GCN-DAG: v_cndmask_b32_e64 v{{[0-9]+}}, 0, v[[LOADHI]], [[BOUNDSMASK]]

define amdgpu_kernel void @bounds_check_load_i64(i64 addrspace(1)* %out, i32 %offset) {
entry:
  %scratch = alloca [4096 x i64], addrspace(5)

  %ptr = getelementptr [4096 x i64], [4096 x i64] addrspace(5)* %scratch, i32 0, i32 %offset
  %value = load i64, i64 addrspace(5)* %ptr
  store i64 %value, i64 addrspace(1)* %out

  ret void
}

; GCN-LABEL: {{^}}bounds_check_store_i64:
; GCN: s_mov_b32 [[BOUNDS:s[0-9]+]], 0x8008
; GFX9: s_and_saveexec_b64 [[EXECMASK:s\[[0-9]+:[0-9]+\]]], [[BOUNDSMASK:s\[[0-9]+:[0-9]+\]]]
; GFX10: s_and_saveexec_b32 [[EXECMASK:s[0-9]+]], [[BOUNDSMASK:s[0-9]+]]
; GCN: buffer_store_dwordx2 v[{{[0-9]+}}:{{[0-9]+}}], [[OFFSET:v[0-9]+]], s[{{[0-9]+}}:{{[0-9]+}}], 0 offen{{$}}
; GFX9: s_mov_b64 exec, [[EXECMASK]]
; GFX10: s_mov_b32 exec_lo, [[EXECMASK]]

define amdgpu_kernel void @bounds_check_store_i64(i64 addrspace(1)* %out, i64 %value, i32 %offset) {
entry:
  %scratch = alloca [4096 x i64], addrspace(5)

  %ptr = getelementptr [4096 x i64], [4096 x i64] addrspace(5)* %scratch, i32 0, i32 %offset
  store i64 %value, i64 addrspace(5)* %ptr
  store i64 %value, i64 addrspace(1)* %out

  ret void
}

; GCN-LABEL: {{^}}bounds_check_load_i128:
; GCN: s_mov_b32 [[BOUNDS:s[0-9]+]], 0x8008
; GFX9: s_and_saveexec_b64 [[EXECMASK:s\[[0-9]+:[0-9]+\]]], [[BOUNDSMASK:s\[[0-9]+:[0-9]+\]]]
; GFX10: s_and_saveexec_b32 [[EXECMASK:s[0-9]+]], [[BOUNDSMASK:s[0-9]+]]
; GCN: buffer_load_dwordx4 v{{\[}}[[LOADLO:[0-9]+]]:[[LOADHI:[0-9]+]]{{\]}}, [[OFFSET:v[0-9]+]], s[{{[0-9]+}}:{{[0-9]+}}], 0 offen{{$}}
; GFX9: s_mov_b64 exec, [[EXECMASK]]
; GFX10: s_mov_b32 exec_lo, [[EXECMASK]]
; GCN-DAG: v_cndmask_b32_e64 v{{[0-9]+}}, 0, v[[LOADLO]], [[BOUNDSMASK]]
; GCN-DAG: v_cndmask_b32_e64 v{{[0-9]+}}, 0, v[[LOADHI]], [[BOUNDSMASK]]

define amdgpu_kernel void @bounds_check_load_i128(i128 addrspace(1)* %out, i32 %offset) {
entry:
  %scratch = alloca [2048 x i128], addrspace(5)

  %ptr = getelementptr [2048 x i128], [2048 x i128] addrspace(5)* %scratch, i32 0, i32 %offset
  %value = load i128, i128 addrspace(5)* %ptr
  store i128 %value, i128 addrspace(1)* %out

  ret void
}

; GCN-LABEL: {{^}}bounds_check_store_i128:
; GCN: s_mov_b32 [[BOUNDS:s[0-9]+]], 0x8008
; GFX9: s_and_saveexec_b64 [[EXECMASK:s\[[0-9]+:[0-9]+\]]], [[BOUNDSMASK:s\[[0-9]+:[0-9]+\]]]
; GFX10: s_and_saveexec_b32 [[EXECMASK:s[0-9]+]], [[BOUNDSMASK:s[0-9]+]]
; GCN: buffer_store_dwordx4 v[{{[0-9]+}}:{{[0-9]+}}], [[OFFSET:v[0-9]+]], s[{{[0-9]+}}:{{[0-9]+}}], 0 offen{{$}}
; GFX9: s_mov_b64 exec, [[EXECMASK]]
; GFX10: s_mov_b32 exec_lo, [[EXECMASK]]

define amdgpu_kernel void @bounds_check_store_i128(i128 addrspace(1)* %out, i128 %value, i32 %offset) {
entry:
  %scratch = alloca [2048 x i128], addrspace(5)

  %ptr = getelementptr [2048 x i128], [2048 x i128] addrspace(5)* %scratch, i32 0, i32 %offset
  store i128 %value, i128 addrspace(5)* %ptr
  store i128 %value, i128 addrspace(1)* %out

  ret void
}

; GCN-LABEL: {{^}}bounds_check_static_valid_store_i32:
; GFX9-NOT: s_and_saveexec_b64
; GFX10-NOT: s_and_saveexec_b32
; GCN: buffer_store_dword v{{[0-9]+}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:20

define amdgpu_kernel void @bounds_check_static_valid_store_i32(i32 addrspace(1)* %out, i32 %value, i32 %offset) {
entry:
  %scratch = alloca [256 x i32], addrspace(5)

  %ptr = getelementptr [256 x i32], [256 x i32] addrspace(5)* %scratch, i32 0, i32 4
  store i32 %value, i32 addrspace(5)* %ptr

  %load_ptr = getelementptr [256 x i32], [256 x i32] addrspace(5)* %scratch, i32 0, i32 %offset
  %val = load i32, i32 addrspace(5)* %load_ptr
  store i32 %val, i32 addrspace(1)* %out

  ret void
}

; GCN-LABEL: {{^}}bounds_check_static_oob_store_i32:
; GCN: buffer_store_dword v{{[0-9]+}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:2052

define amdgpu_kernel void @bounds_check_static_oob_store_i32(i32 addrspace(1)* %out, i32 %value, i32 %offset) {
entry:
  %scratch = alloca [256 x i32], addrspace(5)

  %ptr = getelementptr [256 x i32], [256 x i32] addrspace(5)* %scratch, i32 0, i32 512
  store i32 %value, i32 addrspace(5)* %ptr

  %load_ptr = getelementptr [256 x i32], [256 x i32] addrspace(5)* %scratch, i32 0, i32 %offset
  %val = load i32, i32 addrspace(5)* %load_ptr
  store i32 %val, i32 addrspace(1)* %out

  ret void
}

; GCN-LABEL: {{^}}bounds_check_static_valid_load_i32:
; GCN: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0 offen{{$}}
; GFX9-NOT: s_and_saveexec_b64
; GFX10-NOT: s_and_saveexec_b32
; GCN: buffer_load_dword v{{[0-9]+}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:20
; GCN-NOT: v_cndmask_b32_e64

define amdgpu_kernel void @bounds_check_static_valid_load_i32(i32 addrspace(1)* %out, i32 %value, i32 %offset) {
entry:
  %scratch = alloca [256 x i32], addrspace(5)

  %store_ptr = getelementptr [256 x i32], [256 x i32] addrspace(5)* %scratch, i32 0, i32 %offset
  store i32 %value, i32 addrspace(5)* %store_ptr

  %ptr = getelementptr [256 x i32], [256 x i32] addrspace(5)* %scratch, i32 0, i32 4
  %val = load i32, i32 addrspace(5)* %ptr

  store i32 %val, i32 addrspace(1)* %out

  ret void
}

; GCN-LABEL: {{^}}bounds_check_static_oob_load_i32:
; GCN: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0 offen{{$}}
; GCN: buffer_load_dword v{{[0-9]+}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:2052

define amdgpu_kernel void @bounds_check_static_oob_load_i32(i32 addrspace(1)* %out, i32 %value, i32 %offset) {
entry:
  %scratch = alloca [256 x i32], addrspace(5)

  %store_ptr = getelementptr [256 x i32], [256 x i32] addrspace(5)* %scratch, i32 0, i32 %offset
  store i32 %value, i32 addrspace(5)* %store_ptr

  %ptr = getelementptr [256 x i32], [256 x i32] addrspace(5)* %scratch, i32 0, i32 512
  %val = load i32, i32 addrspace(5)* %ptr

  store i32 %val, i32 addrspace(1)* %out

  ret void
}

; GCN-LABEL: {{^}}bounds_check_load_offset_i32:
; GCN: s_mov_b32 [[BOUNDS:s[0-9]+]], 0x8004
; GFX9: v_add_u32_e32 [[CMPOFFSET:v[0-9]+]], 16, [[OFFSET:v[0-9]+]]
; GFX9: v_cmp_gt_u32_e64 [[BOUNDSMASK:s\[[0-9]+:[0-9]+\]]], [[BOUNDS]], [[CMPOFFSET]]
; GFX9: s_and_saveexec_b64 [[EXECMASK:s\[[0-9]+:[0-9]+\]]], [[BOUNDSMASK]]
; GFX10: v_add_nc_u32_e32 [[CMPOFFSET:v[0-9]+]], 16, [[OFFSET:v[0-9]+]]
; GFX10: v_cmp_gt_u32_e64 [[BOUNDSMASK:s[0-9]+]], [[BOUNDS]], [[CMPOFFSET]]
; GFX10: s_and_saveexec_b32 [[EXECMASK:s[0-9]+]], [[BOUNDSMASK]]
; GCN: buffer_load_dword [[LOADVALUE:v[0-9]+]], [[OFFSET]], s[{{[0-9]+}}:{{[0-9]+}}], 0 offen offset:16{{$}}
; GFX9: s_mov_b64 exec, [[EXECMASK]]
; GFX10: s_mov_b32 exec_lo, [[EXECMASK]]
; GCN: v_cndmask_b32_e64 v{{[0-9]+}}, 0, [[LOADVALUE]], [[BOUNDSMASK]]

define amdgpu_kernel void @bounds_check_load_offset_i32(i32 addrspace(1)* %out, i32 %offset) {
entry:
  %scratch = alloca [8192 x i32], addrspace(5)

  %ptr.0 = getelementptr [8192 x i32], [8192 x i32] addrspace(5)* %scratch, i32 0, i32 %offset
  %ptr.1 = getelementptr i32, i32 addrspace(5)* %ptr.0, i32 4
  %value = load i32, i32 addrspace(5)* %ptr.1
  store i32 %value, i32 addrspace(1)* %out

  ret void
}

; GCN-LABEL: {{^}}bounds_check_store_offset_i32:
; GCN: s_mov_b32 [[BOUNDS:s[0-9]+]], 0x8004
; GFX9: v_add_u32_e32 [[CMPOFFSET:v[0-9]+]], 16, [[OFFSET:v[0-9]+]]
; GFX9: v_cmp_gt_u32_e64 [[BOUNDSMASK:s\[[0-9]+:[0-9]+\]]], [[BOUNDS]], [[CMPOFFSET]]
; GFX9: s_and_saveexec_b64 [[EXECMASK:s\[[0-9]+:[0-9]+\]]], [[BOUNDSMASK]]
; GFX10: v_add_nc_u32_e32 [[CMPOFFSET:v[0-9]+]], 16, [[OFFSET:v[0-9]+]]
; GFX10: v_cmp_gt_u32_e64 [[BOUNDSMASK:s[0-9]+]], [[BOUNDS]], [[CMPOFFSET]]
; GFX10: s_and_saveexec_b32 [[EXECMASK:s[0-9]+]], [[BOUNDSMASK]]
; GCN: buffer_store_dword v{{[0-9]+}}, [[OFFSET]], s[{{[0-9]+}}:{{[0-9]+}}], 0 offen offset:16{{$}}
; GFX9: s_mov_b64 exec, [[EXECMASK]]
; GFX10: s_mov_b32 exec_lo, [[EXECMASK]]

define amdgpu_kernel void @bounds_check_store_offset_i32(i32 addrspace(1)* %out, i32 %value, i32 %offset) {
entry:
  %scratch = alloca [8192 x i32], addrspace(5)

  %ptr.0 = getelementptr [8192 x i32], [8192 x i32] addrspace(5)* %scratch, i32 0, i32 %offset
  %ptr.1 = getelementptr i32, i32 addrspace(5)* %ptr.0, i32 4
  store i32 %value, i32 addrspace(5)* %ptr.1
  store i32 %value, i32 addrspace(1)* %out

  ret void
}
