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

; GCN-LABEL: {{^}}block_split_with_vcc:
;
; Mostly just check this example compile at all.
;
; GCN: s_mov_b32 [[BOUNDS:s[0-9]+]], 0x404
; GFX9: v_cmp_gt_u32_e64 [[BOUNDSMASK:s\[[0-9]+:[0-9]+\]]], [[BOUNDS]], [[OFFSET:v[0-9]+]]
; GFX9: s_and_saveexec_b64 [[EXECMASK:s\[[0-9]+:[0-9]+\]]], [[BOUNDSMASK]]
; GFX10: v_cmp_gt_u32_e64 [[BOUNDSMASK:s[0-9]+]], [[BOUNDS]], [[OFFSET:v[0-9]+]]
; GFX10: s_and_saveexec_b32 [[EXECMASK:s[0-9]+]], [[BOUNDSMASK]]
; GCN: buffer_store_dword [[LOADVALUE:v[0-9]+]], [[OFFSET]], s[{{[0-9]+}}:{{[0-9]+}}]
; GFX9: s_mov_b64 exec, [[EXECMASK]]
; GFX10: s_mov_b32 exec_lo, [[EXECMASK]]

define amdgpu_ps void @block_split_with_vcc(i32 inreg %0, i32 inreg %1, i32 inreg %2, <2 x float> %3, <2 x float> %4, <2 x float> %5, <3 x float> %6, <2 x float> %7, <2 x float> %8, <2 x float> %9, float %10, float %11, float %12, float %13, float %14, i32 %15, i32 %16, i32 %17, i32 %18) local_unnamed_addr {
.entry:
  %19 = alloca [256 x i32], align 4, addrspace(5)
  br label %21

.preheader:                                       ; preds = %21
  %20 = icmp eq i32 %22, 0
  br label %27

21:                                               ; preds = %.entry, %21
  %22 = phi i32 [ 0, %.entry ], [ %25, %21 ]
  %23 = phi i1 [ true, %.entry ], [ %26, %21 ]
  %24 = getelementptr [256 x i32], [256 x i32] addrspace(5)* %19, i32 0, i32 %22
  store i32 0, i32 addrspace(5)* %24, align 4
  %25 = add i32 %22, 1
  %26 = icmp slt i32 %22, 256
  br i1 %23, label %21, label %.preheader, !llvm.loop !1

27:                                               ; preds = %.preheader, %27
  %.1 = phi i32 [ %.i1, %27 ], [ 0, %.preheader ]
  %28 = add nuw nsw i32 %.1, 32
  %29 = getelementptr [256 x i32], [256 x i32] addrspace(5)* %19, i32 0, i32 %28
  %30 = load i32, i32 addrspace(5)* %29, align 4
  %31 = icmp eq i32 %30, 0
  %.i1 = select i1 %31, i32 0, i32 32
  %.not = xor i1 %31, true
  %brmerge = or i1 %20, %.not
  br i1 %brmerge, label %32, label %27

32:                                               ; preds = %27
  call void @llvm.amdgcn.exp.f32(i32 0, i32 1, float 0.000000e+00, float undef, float undef, float undef, i1 true, i1 true)
  ret void
}

declare void @llvm.amdgcn.exp.f32(i32 immarg, i32 immarg, float, float, float, float, i1 immarg, i1 immarg) #0

attributes #0 = { inaccessiblememonly nounwind willreturn writeonly }
attributes #1 = { nounwind "InitialPSInputAddr"="0" "amdgpu-unroll-threshold"="700" "denormal-fp-math-f32"="preserve-sign" "target-features"=",+enable-scratch-bounds-checks" }

!1 = distinct !{!1}
