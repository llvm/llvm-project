; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs -enable-misched -enable-aa-sched-mi < %s | FileCheck -check-prefix=FUNC -check-prefix=CI %s

declare void @llvm.SI.tbuffer.store.i32(<16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
declare void @llvm.SI.tbuffer.store.v4i32(<16 x i8>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
declare void @llvm.amdgcn.s.barrier() #1
declare i32 @llvm.amdgcn.workitem.id.x() #2


@stored_lds_ptr = addrspace(3) global i32 addrspace(3)* undef, align 4
@stored_constant_ptr = addrspace(3) global i32 addrspace(2)* undef, align 8
@stored_global_ptr = addrspace(3) global i32 addrspace(1)* undef, align 8

; FUNC-LABEL: @reorder_local_load_global_store_local_load
; CI: ds_read2_b32 {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}} offset0:1 offset1:3
; CI: buffer_store_dword
define void @reorder_local_load_global_store_local_load(i32 addrspace(1)* %out, i32 addrspace(1)* %gptr) #0 {
  %ptr0 = load i32 addrspace(3)*, i32 addrspace(3)* addrspace(3)* @stored_lds_ptr, align 4

  %ptr1 = getelementptr inbounds i32, i32 addrspace(3)* %ptr0, i32 1
  %ptr2 = getelementptr inbounds i32, i32 addrspace(3)* %ptr0, i32 3

  %tmp1 = load i32, i32 addrspace(3)* %ptr1, align 4
  store i32 99, i32 addrspace(1)* %gptr, align 4
  %tmp2 = load i32, i32 addrspace(3)* %ptr2, align 4

  %add = add nsw i32 %tmp1, %tmp2

  store i32 %add, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @no_reorder_local_load_volatile_global_store_local_load
; CI: ds_read_b32 {{v[0-9]+}}, {{v[0-9]+}} offset:4
; CI: buffer_store_dword
; CI: ds_read_b32 {{v[0-9]+}}, {{v[0-9]+}} offset:12
define void @no_reorder_local_load_volatile_global_store_local_load(i32 addrspace(1)* %out, i32 addrspace(1)* %gptr) #0 {
  %ptr0 = load i32 addrspace(3)*, i32 addrspace(3)* addrspace(3)* @stored_lds_ptr, align 4

  %ptr1 = getelementptr inbounds i32, i32 addrspace(3)* %ptr0, i32 1
  %ptr2 = getelementptr inbounds i32, i32 addrspace(3)* %ptr0, i32 3

  %tmp1 = load i32, i32 addrspace(3)* %ptr1, align 4
  store volatile i32 99, i32 addrspace(1)* %gptr, align 4
  %tmp2 = load i32, i32 addrspace(3)* %ptr2, align 4

  %add = add nsw i32 %tmp1, %tmp2

  store i32 %add, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @no_reorder_barrier_local_load_global_store_local_load
; CI: ds_read_b32 {{v[0-9]+}}, {{v[0-9]+}} offset:4
; CI: ds_read_b32 {{v[0-9]+}}, {{v[0-9]+}} offset:12
; CI: buffer_store_dword
define void @no_reorder_barrier_local_load_global_store_local_load(i32 addrspace(1)* %out, i32 addrspace(1)* %gptr) #0 {
  %ptr0 = load i32 addrspace(3)*, i32 addrspace(3)* addrspace(3)* @stored_lds_ptr, align 4

  %ptr1 = getelementptr inbounds i32, i32 addrspace(3)* %ptr0, i32 1
  %ptr2 = getelementptr inbounds i32, i32 addrspace(3)* %ptr0, i32 3

  %tmp1 = load i32, i32 addrspace(3)* %ptr1, align 4
  store i32 99, i32 addrspace(1)* %gptr, align 4
  call void @llvm.amdgcn.s.barrier() #1
  %tmp2 = load i32, i32 addrspace(3)* %ptr2, align 4

  %add = add nsw i32 %tmp1, %tmp2

  store i32 %add, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @reorder_constant_load_global_store_constant_load
; CI-DAG: v_readfirstlane_b32 s[[PTR_LO:[0-9]+]], v{{[0-9]+}}
; CI: v_readfirstlane_b32 s[[PTR_HI:[0-9]+]], v{{[0-9]+}}
; CI: buffer_store_dword
; CI-DAG: s_load_dword s{{[0-9]+}}, s{{\[}}[[PTR_LO]]:[[PTR_HI]]{{\]}}, 0x1
; CI-DAG: s_load_dword s{{[0-9]+}}, s{{\[}}[[PTR_LO]]:[[PTR_HI]]{{\]}}, 0x3
; CI: buffer_store_dword
define void @reorder_constant_load_global_store_constant_load(i32 addrspace(1)* %out, i32 addrspace(1)* %gptr) #0 {
  %ptr0 = load i32 addrspace(2)*, i32 addrspace(2)* addrspace(3)* @stored_constant_ptr, align 8

  %ptr1 = getelementptr inbounds i32, i32 addrspace(2)* %ptr0, i64 1
  %ptr2 = getelementptr inbounds i32, i32 addrspace(2)* %ptr0, i64 3

  %tmp1 = load i32, i32 addrspace(2)* %ptr1, align 4
  store i32 99, i32 addrspace(1)* %gptr, align 4
  %tmp2 = load i32, i32 addrspace(2)* %ptr2, align 4

  %add = add nsw i32 %tmp1, %tmp2

  store i32 %add, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @reorder_constant_load_local_store_constant_load
; CI: v_readfirstlane_b32 s[[PTR_LO:[0-9]+]], v{{[0-9]+}}
; CI: v_readfirstlane_b32 s[[PTR_HI:[0-9]+]], v{{[0-9]+}}
; CI-DAG: s_load_dword s{{[0-9]+}}, s{{\[}}[[PTR_LO]]:[[PTR_HI]]{{\]}}, 0x1
; CI-DAG: s_load_dword s{{[0-9]+}}, s{{\[}}[[PTR_LO]]:[[PTR_HI]]{{\]}}, 0x3
; CI: ds_write_b32
; CI: buffer_store_dword
define void @reorder_constant_load_local_store_constant_load(i32 addrspace(1)* %out, i32 addrspace(3)* %lptr) #0 {
  %ptr0 = load i32 addrspace(2)*, i32 addrspace(2)* addrspace(3)* @stored_constant_ptr, align 8

  %ptr1 = getelementptr inbounds i32, i32 addrspace(2)* %ptr0, i64 1
  %ptr2 = getelementptr inbounds i32, i32 addrspace(2)* %ptr0, i64 3

  %tmp1 = load i32, i32 addrspace(2)* %ptr1, align 4
  store i32 99, i32 addrspace(3)* %lptr, align 4
  %tmp2 = load i32, i32 addrspace(2)* %ptr2, align 4

  %add = add nsw i32 %tmp1, %tmp2

  store i32 %add, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @reorder_smrd_load_local_store_smrd_load
; CI: s_load_dword
; CI: s_load_dword
; CI: s_load_dword
; CI: ds_write_b32
; CI: buffer_store_dword
define void @reorder_smrd_load_local_store_smrd_load(i32 addrspace(1)* %out, i32 addrspace(3)* noalias %lptr, i32 addrspace(2)* %ptr0) #0 {
  %ptr1 = getelementptr inbounds i32, i32 addrspace(2)* %ptr0, i64 1
  %ptr2 = getelementptr inbounds i32, i32 addrspace(2)* %ptr0, i64 2

  %tmp1 = load i32, i32 addrspace(2)* %ptr1, align 4
  store i32 99, i32 addrspace(3)* %lptr, align 4
  %tmp2 = load i32, i32 addrspace(2)* %ptr2, align 4

  %add = add nsw i32 %tmp1, %tmp2

  store i32 %add, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @reorder_global_load_local_store_global_load
; CI: ds_write_b32
; CI: buffer_load_dword
; CI: buffer_load_dword
; CI: buffer_store_dword
define void @reorder_global_load_local_store_global_load(i32 addrspace(1)* %out, i32 addrspace(3)* %lptr, i32 addrspace(1)* %ptr0) #0 {
  %ptr1 = getelementptr inbounds i32, i32 addrspace(1)* %ptr0, i64 1
  %ptr2 = getelementptr inbounds i32, i32 addrspace(1)* %ptr0, i64 3

  %tmp1 = load i32, i32 addrspace(1)* %ptr1, align 4
  store i32 99, i32 addrspace(3)* %lptr, align 4
  %tmp2 = load i32, i32 addrspace(1)* %ptr2, align 4

  %add = add nsw i32 %tmp1, %tmp2

  store i32 %add, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @reorder_local_offsets
; CI: ds_read2_b32 {{v\[[0-9]+:[0-9]+\]}}, {{v[0-9]+}} offset0:100 offset1:102
; CI-DAG: ds_write2_b32 {{v[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset0:3 offset1:100
; CI-DAG: ds_write_b32 {{v[0-9]+}}, {{v[0-9]+}} offset:408
; CI: buffer_store_dword
; CI: s_endpgm
define void @reorder_local_offsets(i32 addrspace(1)* nocapture %out, i32 addrspace(1)* noalias nocapture readnone %gptr, i32 addrspace(3)* noalias nocapture %ptr0) #0 {
  %ptr1 = getelementptr inbounds i32, i32 addrspace(3)* %ptr0, i32 3
  %ptr2 = getelementptr inbounds i32, i32 addrspace(3)* %ptr0, i32 100
  %ptr3 = getelementptr inbounds i32, i32 addrspace(3)* %ptr0, i32 102

  store i32 123, i32 addrspace(3)* %ptr1, align 4
  %tmp1 = load i32, i32 addrspace(3)* %ptr2, align 4
  %tmp2 = load i32, i32 addrspace(3)* %ptr3, align 4
  store i32 123, i32 addrspace(3)* %ptr2, align 4
  %tmp3 = load i32, i32 addrspace(3)* %ptr1, align 4
  store i32 789, i32 addrspace(3)* %ptr3, align 4

  %add.0 = add nsw i32 %tmp2, %tmp1
  %add.1 = add nsw i32 %add.0, %tmp3
  store i32 %add.1, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @reorder_global_offsets
; CI-DAG: buffer_load_dword {{v[0-9]+}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 offset:400
; CI-DAG: buffer_load_dword {{v[0-9]+}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 offset:408
; CI-DAG: buffer_store_dword {{v[0-9]+}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 offset:12
; CI-DAG: buffer_store_dword {{v[0-9]+}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 offset:400
; CI-DAG: buffer_store_dword {{v[0-9]+}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 offset:408
; CI: buffer_store_dword
; CI: s_endpgm
define void @reorder_global_offsets(i32 addrspace(1)* nocapture %out, i32 addrspace(1)* noalias nocapture readnone %gptr, i32 addrspace(1)* noalias nocapture %ptr0) #0 {
  %ptr1 = getelementptr inbounds i32, i32 addrspace(1)* %ptr0, i32 3
  %ptr2 = getelementptr inbounds i32, i32 addrspace(1)* %ptr0, i32 100
  %ptr3 = getelementptr inbounds i32, i32 addrspace(1)* %ptr0, i32 102

  store i32 123, i32 addrspace(1)* %ptr1, align 4
  %tmp1 = load i32, i32 addrspace(1)* %ptr2, align 4
  %tmp2 = load i32, i32 addrspace(1)* %ptr3, align 4
  store i32 123, i32 addrspace(1)* %ptr2, align 4
  %tmp3 = load i32, i32 addrspace(1)* %ptr1, align 4
  store i32 789, i32 addrspace(1)* %ptr3, align 4

  %add.0 = add nsw i32 %tmp2, %tmp1
  %add.1 = add nsw i32 %add.0, %tmp3
  store i32 %add.1, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}reorder_global_offsets_addr64_soffset0:
; GCN: buffer_load_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} 0 addr64 offset:12{{$}}
; GCN-NEXT: buffer_load_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} 0 addr64 offset:28{{$}}
; GCN-NEXT: buffer_load_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} 0 addr64 offset:44{{$}}

; GCN: v_mov_b32
; GCN: v_mov_b32

; GCN: buffer_store_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} 0 addr64{{$}}
; GCN-NEXT: buffer_store_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} 0 addr64 offset:20{{$}}

; GCN: v_add_i32
; GCN: v_add_i32

; GCN: buffer_store_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} 0 addr64 offset:36{{$}}
; GCN-NEXT: buffer_store_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}} 0 addr64 offset:52{{$}}
define void @reorder_global_offsets_addr64_soffset0(i32 addrspace(1)* noalias nocapture %ptr.base) #0 {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %id.ext = sext i32 %id to i64

  %ptr0 = getelementptr inbounds i32, i32 addrspace(1)* %ptr.base, i64 %id.ext
  %ptr1 = getelementptr inbounds i32, i32 addrspace(1)* %ptr0, i32 3
  %ptr2 = getelementptr inbounds i32, i32 addrspace(1)* %ptr0, i32 5
  %ptr3 = getelementptr inbounds i32, i32 addrspace(1)* %ptr0, i32 7
  %ptr4 = getelementptr inbounds i32, i32 addrspace(1)* %ptr0, i32 9
  %ptr5 = getelementptr inbounds i32, i32 addrspace(1)* %ptr0, i32 11
  %ptr6 = getelementptr inbounds i32, i32 addrspace(1)* %ptr0, i32 13

  store i32 789, i32 addrspace(1)* %ptr0, align 4
  %tmp1 = load i32, i32 addrspace(1)* %ptr1, align 4
  store i32 123, i32 addrspace(1)* %ptr2, align 4
  %tmp2 = load i32, i32 addrspace(1)* %ptr3, align 4
  %add.0 = add nsw i32 %tmp1, %tmp2
  store i32 %add.0, i32 addrspace(1)* %ptr4, align 4
  %tmp3 = load i32, i32 addrspace(1)* %ptr5, align 4
  %add.1 = add nsw i32 %add.0, %tmp3
  store i32 %add.1, i32 addrspace(1)* %ptr6, align 4
  ret void
}

; XFUNC-LABEL: @reorder_local_load_tbuffer_store_local_load
; XCI: ds_read_b32 {{v[0-9]+}}, {{v[0-9]+}}, 0x4
; XCI: TBUFFER_STORE_FORMAT
; XCI: ds_read_b32 {{v[0-9]+}}, {{v[0-9]+}}, 0x8
; define amdgpu_vs void @reorder_local_load_tbuffer_store_local_load(i32 addrspace(1)* %out, i32 %a1, i32 %vaddr) #0 {
;   %ptr0 = load i32 addrspace(3)*, i32 addrspace(3)* addrspace(3)* @stored_lds_ptr, align 4

;   %ptr1 = getelementptr inbounds i32, i32 addrspace(3)* %ptr0, i32 1
;   %ptr2 = getelementptr inbounds i32, i32 addrspace(3)* %ptr0, i32 2

;   %tmp1 = load i32, i32 addrspace(3)* %ptr1, align 4

;   %vdata = insertelement <4 x i32> undef, i32 %a1, i32 0
;   call void @llvm.SI.tbuffer.store.v4i32(<16 x i8> undef, <4 x i32> %vdata,
;         i32 4, i32 %vaddr, i32 0, i32 32, i32 14, i32 4, i32 1, i32 0, i32 1,
;         i32 1, i32 0)

;   %tmp2 = load i32, i32 addrspace(3)* %ptr2, align 4

;   %add = add nsw i32 %tmp1, %tmp2

;   store i32 %add, i32 addrspace(1)* %out, align 4
;   ret void
; }

attributes #0 = { nounwind }
attributes #1 = { nounwind convergent }
attributes #2 = { nounwind readnone }
