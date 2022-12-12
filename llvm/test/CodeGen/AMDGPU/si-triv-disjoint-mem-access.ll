; RUN: llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=bonaire -enable-amdgpu-aa=0 -verify-machineinstrs -enable-misched -enable-aa-sched-mi < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CI %s
; RUN: llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=gfx900 -enable-amdgpu-aa=0 -verify-machineinstrs -enable-misched -enable-aa-sched-mi < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9 %s

@stored_lds_ptr = addrspace(3) global ptr addrspace(3) undef, align 4
@stored_constant_ptr = addrspace(3) global ptr addrspace(4) undef, align 8
@stored_global_ptr = addrspace(3) global ptr addrspace(1) undef, align 8

; GCN-LABEL: {{^}}reorder_local_load_global_store_local_load:
; CI: ds_read2_b32 {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}} offset0:1 offset1:3
; CI: buffer_store_dword

; GFX9: ds_read2_b32 {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}} offset0:1 offset1:3
; GFX9: global_store_dword
; GFX9: global_store_dword
define amdgpu_kernel void @reorder_local_load_global_store_local_load(ptr addrspace(1) %out, ptr addrspace(1) %gptr) #0 {
  %ptr0 = load ptr addrspace(3), ptr addrspace(3) @stored_lds_ptr, align 4

  %ptr1 = getelementptr inbounds i32, ptr addrspace(3) %ptr0, i32 1
  %ptr2 = getelementptr inbounds i32, ptr addrspace(3) %ptr0, i32 3

  %tmp1 = load i32, ptr addrspace(3) %ptr1, align 4
  store i32 99, ptr addrspace(1) %gptr, align 4
  %tmp2 = load i32, ptr addrspace(3) %ptr2, align 4

  %add = add nsw i32 %tmp1, %tmp2

  store i32 %add, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}no_reorder_local_load_volatile_global_store_local_load:
; CI: ds_read_b32 {{v[0-9]+}}, {{v[0-9]+}} offset:4
; CI: buffer_store_dword
; CI: ds_read_b32 {{v[0-9]+}}, {{v[0-9]+}} offset:12

; GFX9: ds_read_b32 {{v[0-9]+}}, {{v[0-9]+}} offset:4
; GFX9: global_store_dword
; GFX9: ds_read_b32 {{v[0-9]+}}, {{v[0-9]+}} offset:12
define amdgpu_kernel void @no_reorder_local_load_volatile_global_store_local_load(ptr addrspace(1) %out, ptr addrspace(1) %gptr) #0 {
  %ptr0 = load ptr addrspace(3), ptr addrspace(3) @stored_lds_ptr, align 4

  %ptr1 = getelementptr inbounds i32, ptr addrspace(3) %ptr0, i32 1
  %ptr2 = getelementptr inbounds i32, ptr addrspace(3) %ptr0, i32 3

  %tmp1 = load i32, ptr addrspace(3) %ptr1, align 4
  store volatile i32 99, ptr addrspace(1) %gptr, align 4
  %tmp2 = load i32, ptr addrspace(3) %ptr2, align 4

  %add = add nsw i32 %tmp1, %tmp2

  store i32 %add, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}no_reorder_barrier_local_load_global_store_local_load:
; CI: ds_read_b32 {{v[0-9]+}}, {{v[0-9]+}} offset:4
; CI: ds_read_b32 {{v[0-9]+}}, {{v[0-9]+}} offset:12
; CI: buffer_store_dword

; GFX9-DAG: global_store_dword
; GFX9-DAG: ds_read_b32 {{v[0-9]+}}, {{v[0-9]+}} offset:4
; GFX9: s_barrier
; GFX9-DAG: ds_read_b32 {{v[0-9]+}}, {{v[0-9]+}} offset:12
; GFX9-DAG: global_store_dword
define amdgpu_kernel void @no_reorder_barrier_local_load_global_store_local_load(ptr addrspace(1) %out, ptr addrspace(1) %gptr) #0 {
  %ptr0 = load ptr addrspace(3), ptr addrspace(3) @stored_lds_ptr, align 4

  %ptr1 = getelementptr inbounds i32, ptr addrspace(3) %ptr0, i32 1
  %ptr2 = getelementptr inbounds i32, ptr addrspace(3) %ptr0, i32 3

  %tmp1 = load i32, ptr addrspace(3) %ptr1, align 4
  store i32 99, ptr addrspace(1) %gptr, align 4
  call void @llvm.amdgcn.s.barrier() #1
  %tmp2 = load i32, ptr addrspace(3) %ptr2, align 4

  %add = add nsw i32 %tmp1, %tmp2

  store i32 %add, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}reorder_constant_load_global_store_constant_load:
; GCN-DAG: v_readfirstlane_b32 s[[PTR_LO:[0-9]+]], v{{[0-9]+}}
; GCN: v_readfirstlane_b32 s[[PTR_HI:[0-9]+]], v{{[0-9]+}}

; CI: s_load_dword s{{[0-9]+}}, s[[[PTR_LO]]:[[PTR_HI]]], 0x1
; CI: buffer_store_dword
; CI: s_load_dword s{{[0-9]+}}, s[[[PTR_LO]]:[[PTR_HI]]], 0x3

; GFX9: s_load_dword s{{[0-9]+}}, s[[[PTR_LO]]:[[PTR_HI]]], 0x4
; GFX9: global_store_dword
; GFX9: s_load_dword s{{[0-9]+}}, s[[[PTR_LO]]:[[PTR_HI]]], 0xc

; CI: buffer_store_dword
; GFX9: global_store_dword
define amdgpu_kernel void @reorder_constant_load_global_store_constant_load(ptr addrspace(1) %out, ptr addrspace(1) %gptr) #0 {
  %ptr0 = load ptr addrspace(4), ptr addrspace(3) @stored_constant_ptr, align 8

  %ptr1 = getelementptr inbounds i32, ptr addrspace(4) %ptr0, i64 1
  %ptr2 = getelementptr inbounds i32, ptr addrspace(4) %ptr0, i64 3

  %tmp1 = load i32, ptr addrspace(4) %ptr1, align 4
  store i32 99, ptr addrspace(1) %gptr, align 4
  %tmp2 = load i32, ptr addrspace(4) %ptr2, align 4

  %add = add nsw i32 %tmp1, %tmp2

  store i32 %add, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}reorder_constant_load_local_store_constant_load:
; GCN: v_readfirstlane_b32 s[[PTR_LO:[0-9]+]], v{{[0-9]+}}
; GCN: v_readfirstlane_b32 s[[PTR_HI:[0-9]+]], v{{[0-9]+}}

; CI-DAG: s_load_dword s{{[0-9]+}}, s[[[PTR_LO]]:[[PTR_HI]]], 0x1
; CI-DAG: s_load_dword s{{[0-9]+}}, s[[[PTR_LO]]:[[PTR_HI]]], 0x3

; GFX9-DAG: s_load_dword s{{[0-9]+}}, s[[[PTR_LO]]:[[PTR_HI]]], 0x4
; GFX9-DAG: s_load_dword s{{[0-9]+}}, s[[[PTR_LO]]:[[PTR_HI]]], 0xc

; GCN-DAG: ds_write_b32
; CI: buffer_store_dword
; GFX9: global_store_dword
define amdgpu_kernel void @reorder_constant_load_local_store_constant_load(ptr addrspace(1) %out, ptr addrspace(3) %lptr) #0 {
  %ptr0 = load ptr addrspace(4), ptr addrspace(3) @stored_constant_ptr, align 8

  %ptr1 = getelementptr inbounds i32, ptr addrspace(4) %ptr0, i64 1
  %ptr2 = getelementptr inbounds i32, ptr addrspace(4) %ptr0, i64 3

  %tmp1 = load i32, ptr addrspace(4) %ptr1, align 4
  store i32 99, ptr addrspace(3) %lptr, align 4
  %tmp2 = load i32, ptr addrspace(4) %ptr2, align 4

  %add = add nsw i32 %tmp1, %tmp2

  store i32 %add, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}reorder_smrd_load_local_store_smrd_load:
; GCN: s_load_dword
; GCN: s_load_dword
; GCN: s_load_dword
; GCN: ds_write_b32
; CI: buffer_store_dword
; GFX9: global_store_dword
define amdgpu_kernel void @reorder_smrd_load_local_store_smrd_load(ptr addrspace(1) %out, ptr addrspace(3) noalias %lptr, ptr addrspace(4) %ptr0) #0 {
  %ptr1 = getelementptr inbounds i32, ptr addrspace(4) %ptr0, i64 1
  %ptr2 = getelementptr inbounds i32, ptr addrspace(4) %ptr0, i64 2

  %tmp1 = load i32, ptr addrspace(4) %ptr1, align 4
  store i32 99, ptr addrspace(3) %lptr, align 4
  %tmp2 = load i32, ptr addrspace(4) %ptr2, align 4

  %add = add nsw i32 %tmp1, %tmp2

  store i32 %add, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}reorder_global_load_local_store_global_load:
; CI: buffer_load_dword
; CI: buffer_load_dword
; CI: ds_write_b32
; CI: buffer_store_dword

; GFX9: global_load_dword v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}} offset:4
; GFX9: global_load_dword v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}} offset:12
; GFX9: ds_write_b32
define amdgpu_kernel void @reorder_global_load_local_store_global_load(ptr addrspace(1) %out, ptr addrspace(3) %lptr, ptr addrspace(1) %ptr0) #0 {
  %ptr1 = getelementptr inbounds i32, ptr addrspace(1) %ptr0, i64 1
  %ptr2 = getelementptr inbounds i32, ptr addrspace(1) %ptr0, i64 3

  %tmp1 = load i32, ptr addrspace(1) %ptr1, align 4
  store i32 99, ptr addrspace(3) %lptr, align 4
  %tmp2 = load i32, ptr addrspace(1) %ptr2, align 4

  %add = add nsw i32 %tmp1, %tmp2

  store i32 %add, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}reorder_local_offsets:
; GCN: ds_read2_b32 {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}} offset0:100 offset1:102
; GCN-DAG: ds_write2_b32 {{v[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset0:3 offset1:100
; GCN-DAG: ds_write_b32 {{v[0-9]+}}, {{v[0-9]+}} offset:408
; CI: buffer_store_dword
; GFX9: global_store_dword
; GCN: s_endpgm
define amdgpu_kernel void @reorder_local_offsets(ptr addrspace(1) nocapture %out, ptr addrspace(1) noalias nocapture readnone %gptr, ptr addrspace(3) noalias nocapture %ptr0) #0 {
  %ptr1 = getelementptr inbounds i32, ptr addrspace(3) %ptr0, i32 3
  %ptr2 = getelementptr inbounds i32, ptr addrspace(3) %ptr0, i32 100
  %ptr3 = getelementptr inbounds i32, ptr addrspace(3) %ptr0, i32 102

  store i32 123, ptr addrspace(3) %ptr1, align 4
  %tmp1 = load i32, ptr addrspace(3) %ptr2, align 4
  %tmp2 = load i32, ptr addrspace(3) %ptr3, align 4
  store i32 123, ptr addrspace(3) %ptr2, align 4
  %tmp3 = load i32, ptr addrspace(3) %ptr1, align 4
  store i32 789, ptr addrspace(3) %ptr3, align 4

  %add.0 = add nsw i32 %tmp2, %tmp1
  %add.1 = add nsw i32 %add.0, %tmp3
  store i32 %add.1, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}reorder_global_offsets:
; CI-DAG: buffer_load_dword {{v[0-9]+}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 offset:400
; CI-DAG: buffer_load_dword {{v[0-9]+}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 offset:408
; CI-DAG: buffer_store_dword {{v[0-9]+}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 offset:12
; CI-DAG: buffer_store_dword {{v[0-9]+}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 offset:400
; CI-DAG: buffer_store_dword {{v[0-9]+}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 offset:408
; CI: buffer_store_dword
; CI: s_endpgm

; GFX9-DAG: global_load_dword {{v[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}} offset:400
; GFX9-DAG: global_load_dword {{v[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}} offset:408
; GFX9-DAG: global_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}} offset:12
; GFX9-DAG: global_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}} offset:400
; GFX9-DAG: global_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}} offset:408
; GFX9: global_store_dword
; GFX9: s_endpgm
define amdgpu_kernel void @reorder_global_offsets(ptr addrspace(1) nocapture %out, ptr addrspace(1) noalias nocapture readnone %gptr, ptr addrspace(1) noalias nocapture %ptr0) #0 {
  %ptr1 = getelementptr inbounds i32, ptr addrspace(1) %ptr0, i32 3
  %ptr2 = getelementptr inbounds i32, ptr addrspace(1) %ptr0, i32 100
  %ptr3 = getelementptr inbounds i32, ptr addrspace(1) %ptr0, i32 102

  store i32 123, ptr addrspace(1) %ptr1, align 4
  %tmp1 = load i32, ptr addrspace(1) %ptr2, align 4
  %tmp2 = load i32, ptr addrspace(1) %ptr3, align 4
  store i32 123, ptr addrspace(1) %ptr2, align 4
  %tmp3 = load i32, ptr addrspace(1) %ptr1, align 4
  store i32 789, ptr addrspace(1) %ptr3, align 4

  %add.0 = add nsw i32 %tmp2, %tmp1
  %add.1 = add nsw i32 %add.0, %tmp3
  store i32 %add.1, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}reorder_global_offsets_addr64_soffset0:
; CI:      buffer_load_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:12{{$}}
; CI-NEXT: buffer_load_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:28{{$}}
; CI-NEXT: buffer_load_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:44{{$}}

; CI: v_mov_b32
; CI: v_mov_b32

; CI-DAG: v_add_i32
; CI-DAG: v_add_i32

; CI-DAG: buffer_store_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; CI-DAG: buffer_store_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:20{{$}}
; CI-DAG: buffer_store_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:36{{$}}
; CI: buffer_store_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:52{{$}}

; GFX9: global_load_dword {{v[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}} offset:12
; GFX9: global_load_dword {{v[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}} offset:28
; GFX9: global_load_dword {{v[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}} offset:44

; GFX9: global_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]$}}
; GFX9: global_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}} offset:20
; GFX9: global_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}} offset:36
; GFX9: global_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}} offset:52

define amdgpu_kernel void @reorder_global_offsets_addr64_soffset0(ptr addrspace(1) noalias nocapture %ptr.base) #0 {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %id.ext = sext i32 %id to i64

  %ptr0 = getelementptr inbounds i32, ptr addrspace(1) %ptr.base, i64 %id.ext
  %ptr1 = getelementptr inbounds i32, ptr addrspace(1) %ptr0, i32 3
  %ptr2 = getelementptr inbounds i32, ptr addrspace(1) %ptr0, i32 5
  %ptr3 = getelementptr inbounds i32, ptr addrspace(1) %ptr0, i32 7
  %ptr4 = getelementptr inbounds i32, ptr addrspace(1) %ptr0, i32 9
  %ptr5 = getelementptr inbounds i32, ptr addrspace(1) %ptr0, i32 11
  %ptr6 = getelementptr inbounds i32, ptr addrspace(1) %ptr0, i32 13

  store i32 789, ptr addrspace(1) %ptr0, align 4
  %tmp1 = load i32, ptr addrspace(1) %ptr1, align 4
  store i32 123, ptr addrspace(1) %ptr2, align 4
  %tmp2 = load i32, ptr addrspace(1) %ptr3, align 4
  %add.0 = add nsw i32 %tmp1, %tmp2
  store i32 %add.0, ptr addrspace(1) %ptr4, align 4
  %tmp3 = load i32, ptr addrspace(1) %ptr5, align 4
  %add.1 = add nsw i32 %add.0, %tmp3
  store i32 %add.1, ptr addrspace(1) %ptr6, align 4
  ret void
}

; GCN-LABEL: {{^}}reorder_local_load_tbuffer_store_local_load:
; GCN: ds_read2_b32 {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}} offset0:1 offset1:2
; GCN: tbuffer_store_format
define amdgpu_vs void @reorder_local_load_tbuffer_store_local_load(ptr addrspace(1) %out, i32 %a1, i32 %vaddr) #0 {
  %ptr0 = load ptr addrspace(3), ptr addrspace(3) @stored_lds_ptr, align 4

  %ptr1 = getelementptr inbounds i32, ptr addrspace(3) %ptr0, i32 1
  %ptr2 = getelementptr inbounds i32, ptr addrspace(3) %ptr0, i32 2

  %tmp1 = load i32, ptr addrspace(3) %ptr1, align 4

  %vdata = insertelement <4 x i32> undef, i32 %a1, i32 0
  %vaddr.add = add i32 %vaddr, 32
  call void @llvm.amdgcn.struct.tbuffer.store.v4i32(<4 x i32> %vdata, <4 x i32> undef, i32 %vaddr.add, i32 0, i32 0, i32 228, i32 3)

  %tmp2 = load i32, ptr addrspace(3) %ptr2, align 4

  %add = add nsw i32 %tmp1, %tmp2
  store i32 %add, ptr addrspace(1) %out, align 4
  ret void
}

declare void @llvm.amdgcn.s.barrier() #1
declare i32 @llvm.amdgcn.workitem.id.x() #2
declare void @llvm.amdgcn.struct.tbuffer.store.v4i32(<4 x i32>, <4 x i32>, i32, i32, i32, i32 immarg, i32 immarg) #3

attributes #0 = { nounwind }
attributes #1 = { convergent nounwind willreturn }
attributes #2 = { nounwind readnone speculatable willreturn }
attributes #3 = { nounwind willreturn writeonly }
