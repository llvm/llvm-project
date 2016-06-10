; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=CI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

declare i32 @llvm.amdgcn.atomic.inc.i32.p1i32(i32 addrspace(1)* nocapture, i32) #2
declare i32 @llvm.amdgcn.atomic.inc.i32.p3i32(i32 addrspace(3)* nocapture, i32) #2
declare i32 @llvm.amdgcn.atomic.inc.i32.p4i32(i32 addrspace(4)* nocapture, i32) #2

declare i64 @llvm.amdgcn.atomic.inc.i64.p1i64(i64 addrspace(1)* nocapture, i64) #2
declare i64 @llvm.amdgcn.atomic.inc.i64.p3i64(i64 addrspace(3)* nocapture, i64) #2
declare i64 @llvm.amdgcn.atomic.inc.i64.p4i64(i64 addrspace(4)* nocapture, i64) #2

declare i32 @llvm.amdgcn.workitem.id.x() #1

; GCN-LABEL: {{^}}lds_atomic_inc_ret_i32:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 42
; GCN: ds_inc_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, [[K]]
define void @lds_atomic_inc_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) #0 {
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p3i32(i32 addrspace(3)* %ptr, i32 42)
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_inc_ret_i32_offset:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 42
; GCN: ds_inc_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, [[K]] offset:16
define void @lds_atomic_inc_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) #0 {
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 4
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p3i32(i32 addrspace(3)* %gep, i32 42)
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_inc_noret_i32:
; GCN: s_load_dword [[SPTR:s[0-9]+]],
; GCN: v_mov_b32_e32 [[DATA:v[0-9]+]], 4
; GCN: v_mov_b32_e32 [[VPTR:v[0-9]+]], [[SPTR]]
; GCN: ds_inc_u32 [[VPTR]], [[DATA]]
define void @lds_atomic_inc_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p3i32(i32 addrspace(3)* %ptr, i32 42)
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_inc_noret_i32_offset:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 42
; GCN: ds_inc_u32 v{{[0-9]+}}, [[K]] offset:16
define void @lds_atomic_inc_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 4
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p3i32(i32 addrspace(3)* %gep, i32 42)
  ret void
}

; GCN-LABEL: {{^}}global_atomic_inc_ret_i32:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 42
; GCN: buffer_atomic_inc [[K]], off, s{{\[[0-9]+:[0-9]+\]}}, 0 glc{{$}}
define void @global_atomic_inc_ret_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %ptr) #0 {
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p1i32(i32 addrspace(1)* %ptr, i32 42)
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}global_atomic_inc_ret_i32_offset:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 42
; GCN: buffer_atomic_inc [[K]], off, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:16 glc{{$}}
define void @global_atomic_inc_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %ptr) #0 {
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i32 4
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p1i32(i32 addrspace(1)* %gep, i32 42)
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_atomic_inc_noret_i32:
; GCN: buffer_atomic_inc [[K]], off, s{{\[[0-9]+:[0-9]+\]}}, 0{{$}}
define void @global_atomic_inc_noret_i32(i32 addrspace(1)* %ptr) nounwind {
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p1i32(i32 addrspace(1)* %ptr, i32 42)
  ret void
}

; FUNC-LABEL: {{^}}global_atomic_inc_noret_i32_offset:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 42
; GCN: buffer_atomic_inc [[K]], off, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:16{{$}}
define void @global_atomic_inc_noret_i32_offset(i32 addrspace(1)* %ptr) nounwind {
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i32 4
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p1i32(i32 addrspace(1)* %gep, i32 42)
  ret void
}

; GCN-LABEL: {{^}}global_atomic_inc_ret_i32_offset_addr64:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 42
; CI: buffer_atomic_inc [[K]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:20 glc{{$}}
; VI: flat_atomic_inc v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, [[K]] glc{{$}}
define void @global_atomic_inc_ret_i32_offset_addr64(i32 addrspace(1)* %out, i32 addrspace(1)* %ptr) #0 {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.tid = getelementptr i32, i32 addrspace(1)* %ptr, i32 %id
  %out.gep = getelementptr i32, i32 addrspace(1)* %out, i32 %id
  %gep = getelementptr i32, i32 addrspace(1)* %gep.tid, i32 5
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p1i32(i32 addrspace(1)* %gep, i32 42)
  store i32 %result, i32 addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}global_atomic_inc_noret_i32_offset_addr64:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 42
; CI: buffer_atomic_inc [[K]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:20{{$}}
; VI: flat_atomic_inc v{{\[[0-9]+:[0-9]+\]}}, [[K]]{{$}}
define void @global_atomic_inc_noret_i32_offset_addr64(i32 addrspace(1)* %ptr) #0 {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.tid = getelementptr i32, i32 addrspace(1)* %ptr, i32 %id
  %gep = getelementptr i32, i32 addrspace(1)* %gep.tid, i32 5
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p1i32(i32 addrspace(1)* %gep, i32 42)
  ret void
}

@lds0 = addrspace(3) global [512 x i32] undef, align 4

; GCN-LABEL: {{^}}atomic_inc_shl_base_lds_0_i32:
; GCN: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; GCN: ds_inc_rtn_u32 {{v[0-9]+}}, [[PTR]], {{v[0-9]+}} offset:8
define void @atomic_inc_shl_base_lds_0_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x i32], [512 x i32] addrspace(3)* @lds0, i32 0, i32 %idx.0
  %val0 = call i32 @llvm.amdgcn.atomic.inc.i32.p3i32(i32 addrspace(3)* %arrayidx0, i32 9)
  store i32 %idx.0, i32 addrspace(1)* %add_use
  store i32 %val0, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_inc_ret_i64:
; GCN-DAG: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; GCN-DAG: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; GCN: ds_inc_rtn_u64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, v{{\[}}[[KLO]]:[[KHI]]{{\]}}{{$}}
define void @lds_atomic_inc_ret_i64(i64 addrspace(1)* %out, i64 addrspace(3)* %ptr) #0 {
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p3i64(i64 addrspace(3)* %ptr, i64 42)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_inc_ret_i64_offset:
; GCN-DAG: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; GCN-DAG: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; GCN: ds_inc_rtn_u64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, v{{\[}}[[KLO]]:[[KHI]]{{\]}} offset:32
define void @lds_atomic_inc_ret_i64_offset(i64 addrspace(1)* %out, i64 addrspace(3)* %ptr) #0 {
  %gep = getelementptr i64, i64 addrspace(3)* %ptr, i32 4
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p3i64(i64 addrspace(3)* %gep, i64 42)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_inc_noret_i64:
; GCN-DAG: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; GCN-DAG: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; GCN: ds_inc_u64 v{{[0-9]+}}, v{{\[}}[[KLO]]:[[KHI]]{{\]}}{{$}}
define void @lds_atomic_inc_noret_i64(i64 addrspace(3)* %ptr) nounwind {
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p3i64(i64 addrspace(3)* %ptr, i64 42)
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_inc_noret_i64_offset:
; GCN-DAG: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; GCN-DAG: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; GCN: ds_inc_u64 v{{[0-9]+}}, v{{\[}}[[KLO]]:[[KHI]]{{\]}} offset:32{{$}}
define void @lds_atomic_inc_noret_i64_offset(i64 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i64, i64 addrspace(3)* %ptr, i32 4
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p3i64(i64 addrspace(3)* %gep, i64 42)
  ret void
}

; GCN-LABEL: {{^}}global_atomic_inc_ret_i64:
; GCN-DAG: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; GCN-DAG: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; GCN: buffer_atomic_inc_x2 v{{\[}}[[KLO]]:[[KHI]]{{\]}}, off, s{{\[[0-9]+:[0-9]+\]}}, 0 glc{{$}}
define void @global_atomic_inc_ret_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %ptr) #0 {
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p1i64(i64 addrspace(1)* %ptr, i64 42)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}global_atomic_inc_ret_i64_offset:
; GCN-DAG: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; GCN-DAG: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; GCN: buffer_atomic_inc_x2 v{{\[}}[[KLO]]:[[KHI]]{{\]}}, off, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:32 glc{{$}}
define void @global_atomic_inc_ret_i64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %ptr) #0 {
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i32 4
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p1i64(i64 addrspace(1)* %gep, i64 42)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_atomic_inc_noret_i64:
; GCN-DAG: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; GCN-DAG: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; GCN: buffer_atomic_inc_x2 v{{\[}}[[KLO]]:[[KHI]]{{\]}}, off, s{{\[[0-9]+:[0-9]+\]}}, 0{{$}}
define void @global_atomic_inc_noret_i64(i64 addrspace(1)* %ptr) nounwind {
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p1i64(i64 addrspace(1)* %ptr, i64 42)
  ret void
}

; FUNC-LABEL: {{^}}global_atomic_inc_noret_i64_offset:
; GCN-DAG: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; GCN-DAG: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; GCN: buffer_atomic_inc_x2 v{{\[}}[[KLO]]:[[KHI]]{{\]}}, off, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:32{{$}}
define void @global_atomic_inc_noret_i64_offset(i64 addrspace(1)* %ptr) nounwind {
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i32 4
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p1i64(i64 addrspace(1)* %gep, i64 42)
  ret void
}

; GCN-LABEL: {{^}}global_atomic_inc_ret_i64_offset_addr64:
; GCN-DAG: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; GCN-DAG: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; CI: buffer_atomic_inc_x2 v{{\[}}[[KLO]]:[[KHI]]{{\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:40 glc{{$}}
; VI: flat_atomic_inc_x2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[KLO]]:[[KHI]]{{\]}} glc{{$}}
define void @global_atomic_inc_ret_i64_offset_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %ptr) #0 {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.tid = getelementptr i64, i64 addrspace(1)* %ptr, i32 %id
  %out.gep = getelementptr i64, i64 addrspace(1)* %out, i32 %id
  %gep = getelementptr i64, i64 addrspace(1)* %gep.tid, i32 5
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p1i64(i64 addrspace(1)* %gep, i64 42)
  store i64 %result, i64 addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}global_atomic_inc_noret_i64_offset_addr64:
; GCN-DAG: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; GCN-DAG: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; CI: buffer_atomic_inc_x2 v{{\[}}[[KLO]]:[[KHI]]{{\]}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:40{{$}}
; VI: flat_atomic_inc_x2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[KLO]]:[[KHI]]{{\]}}{{$}}
define void @global_atomic_inc_noret_i64_offset_addr64(i64 addrspace(1)* %ptr) #0 {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.tid = getelementptr i64, i64 addrspace(1)* %ptr, i32 %id
  %gep = getelementptr i64, i64 addrspace(1)* %gep.tid, i32 5
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p1i64(i64 addrspace(1)* %gep, i64 42)
  ret void
}

; GCN-LABEL: {{^}}flat_atomic_inc_ret_i32:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 42
; GCN: flat_atomic_inc v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, [[K]] glc{{$}}
define void @flat_atomic_inc_ret_i32(i32 addrspace(4)* %out, i32 addrspace(4)* %ptr) #0 {
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p4i32(i32 addrspace(4)* %ptr, i32 42)
  store i32 %result, i32 addrspace(4)* %out
  ret void
}

; GCN-LABEL: {{^}}flat_atomic_inc_ret_i32_offset:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 42
; GCN: flat_atomic_inc v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, [[K]] glc{{$}}
define void @flat_atomic_inc_ret_i32_offset(i32 addrspace(4)* %out, i32 addrspace(4)* %ptr) #0 {
  %gep = getelementptr i32, i32 addrspace(4)* %ptr, i32 4
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p4i32(i32 addrspace(4)* %gep, i32 42)
  store i32 %result, i32 addrspace(4)* %out
  ret void
}

; FUNC-LABEL: {{^}}flat_atomic_inc_noret_i32:
; GCN: flat_atomic_inc v{{\[[0-9]+:[0-9]+\]}}, [[K]]{{$}}
define void @flat_atomic_inc_noret_i32(i32 addrspace(4)* %ptr) nounwind {
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p4i32(i32 addrspace(4)* %ptr, i32 42)
  ret void
}

; FUNC-LABEL: {{^}}flat_atomic_inc_noret_i32_offset:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 42
; GCN: flat_atomic_inc v{{\[[0-9]+:[0-9]+\]}}, [[K]]{{$}}
define void @flat_atomic_inc_noret_i32_offset(i32 addrspace(4)* %ptr) nounwind {
  %gep = getelementptr i32, i32 addrspace(4)* %ptr, i32 4
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p4i32(i32 addrspace(4)* %gep, i32 42)
  ret void
}

; GCN-LABEL: {{^}}flat_atomic_inc_ret_i32_offset_addr64:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 42
; GCN: flat_atomic_inc v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, [[K]] glc{{$}}
define void @flat_atomic_inc_ret_i32_offset_addr64(i32 addrspace(4)* %out, i32 addrspace(4)* %ptr) #0 {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.tid = getelementptr i32, i32 addrspace(4)* %ptr, i32 %id
  %out.gep = getelementptr i32, i32 addrspace(4)* %out, i32 %id
  %gep = getelementptr i32, i32 addrspace(4)* %gep.tid, i32 5
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p4i32(i32 addrspace(4)* %gep, i32 42)
  store i32 %result, i32 addrspace(4)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}flat_atomic_inc_noret_i32_offset_addr64:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 42
; GCN: flat_atomic_inc v{{\[[0-9]+:[0-9]+\]}}, [[K]]{{$}}
define void @flat_atomic_inc_noret_i32_offset_addr64(i32 addrspace(4)* %ptr) #0 {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.tid = getelementptr i32, i32 addrspace(4)* %ptr, i32 %id
  %gep = getelementptr i32, i32 addrspace(4)* %gep.tid, i32 5
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p4i32(i32 addrspace(4)* %gep, i32 42)
  ret void
}

@lds1 = addrspace(3) global [512 x i64] undef, align 8

; GCN-LABEL: {{^}}atomic_inc_shl_base_lds_0_i64:
; GCN: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 3, {{v[0-9]+}}
; GCN: ds_inc_rtn_u64 v{{\[[0-9]+:[0-9]+\]}}, [[PTR]], v{{\[[0-9]+:[0-9]+\]}} offset:16
define void @atomic_inc_shl_base_lds_0_i64(i64 addrspace(1)* %out, i32 addrspace(1)* %add_use) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x i64], [512 x i64] addrspace(3)* @lds1, i32 0, i32 %idx.0
  %val0 = call i64 @llvm.amdgcn.atomic.inc.i64.p3i64(i64 addrspace(3)* %arrayidx0, i64 9)
  store i32 %idx.0, i32 addrspace(1)* %add_use
  store i64 %val0, i64 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind argmemonly }






; GCN-LABEL: {{^}}flat_atomic_inc_ret_i64:
; GCN-DAG: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; GCN-DAG: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; GCN: flat_atomic_inc_x2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[KLO]]:[[KHI]]{{\]}} glc{{$}}
define void @flat_atomic_inc_ret_i64(i64 addrspace(4)* %out, i64 addrspace(4)* %ptr) #0 {
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p4i64(i64 addrspace(4)* %ptr, i64 42)
  store i64 %result, i64 addrspace(4)* %out
  ret void
}

; GCN-LABEL: {{^}}flat_atomic_inc_ret_i64_offset:
; GCN-DAG: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; GCN-DAG: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; GCN: flat_atomic_inc_x2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[KLO]]:[[KHI]]{{\]}} glc{{$}}
define void @flat_atomic_inc_ret_i64_offset(i64 addrspace(4)* %out, i64 addrspace(4)* %ptr) #0 {
  %gep = getelementptr i64, i64 addrspace(4)* %ptr, i32 4
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p4i64(i64 addrspace(4)* %gep, i64 42)
  store i64 %result, i64 addrspace(4)* %out
  ret void
}

; FUNC-LABEL: {{^}}flat_atomic_inc_noret_i64:
; GCN-DAG: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; GCN-DAG: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; GCN: flat_atomic_inc_x2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[KLO]]:[[KHI]]{{\]$}}
define void @flat_atomic_inc_noret_i64(i64 addrspace(4)* %ptr) nounwind {
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p4i64(i64 addrspace(4)* %ptr, i64 42)
  ret void
}

; FUNC-LABEL: {{^}}flat_atomic_inc_noret_i64_offset:
; GCN-DAG: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; GCN-DAG: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; GCN: flat_atomic_inc_x2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[KLO]]:[[KHI]]{{\]$}}
define void @flat_atomic_inc_noret_i64_offset(i64 addrspace(4)* %ptr) nounwind {
  %gep = getelementptr i64, i64 addrspace(4)* %ptr, i32 4
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p4i64(i64 addrspace(4)* %gep, i64 42)
  ret void
}

; GCN-LABEL: {{^}}flat_atomic_inc_ret_i64_offset_addr64:
; GCN-DAG: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; GCN-DAG: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; GCN: flat_atomic_inc_x2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[KLO]]:[[KHI]]{{\]}} glc{{$}}
define void @flat_atomic_inc_ret_i64_offset_addr64(i64 addrspace(4)* %out, i64 addrspace(4)* %ptr) #0 {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.tid = getelementptr i64, i64 addrspace(4)* %ptr, i32 %id
  %out.gep = getelementptr i64, i64 addrspace(4)* %out, i32 %id
  %gep = getelementptr i64, i64 addrspace(4)* %gep.tid, i32 5
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p4i64(i64 addrspace(4)* %gep, i64 42)
  store i64 %result, i64 addrspace(4)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}flat_atomic_inc_noret_i64_offset_addr64:
; GCN-DAG: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; GCN-DAG: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; GCN: flat_atomic_inc_x2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[KLO]]:[[KHI]]{{\]$}}
define void @flat_atomic_inc_noret_i64_offset_addr64(i64 addrspace(4)* %ptr) #0 {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.tid = getelementptr i64, i64 addrspace(4)* %ptr, i32 %id
  %gep = getelementptr i64, i64 addrspace(4)* %gep.tid, i32 5
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p4i64(i64 addrspace(4)* %gep, i64 42)
  ret void
}
