; RUN: llc -mtriple=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CI,CIVI %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI,CIVI %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9 %s

declare i32 @llvm.amdgcn.atomic.inc.i32.p1(ptr addrspace(1) nocapture, i32, i32, i32, i1) #2
declare i32 @llvm.amdgcn.atomic.inc.i32.p3(ptr addrspace(3) nocapture, i32, i32, i32, i1) #2
declare i32 @llvm.amdgcn.atomic.inc.i32.p0(ptr nocapture, i32, i32, i32, i1) #2

declare i64 @llvm.amdgcn.atomic.inc.i64.p1(ptr addrspace(1) nocapture, i64, i32, i32, i1) #2
declare i64 @llvm.amdgcn.atomic.inc.i64.p3(ptr addrspace(3) nocapture, i64, i32, i32, i1) #2
declare i64 @llvm.amdgcn.atomic.inc.i64.p0(ptr nocapture, i64, i32, i32, i1) #2

declare i32 @llvm.amdgcn.workitem.id.x() #1

; GCN-LABEL: {{^}}lds_atomic_inc_ret_i32:
; CIVI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 42
; GCN: ds_inc_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, [[K]]
define amdgpu_kernel void @lds_atomic_inc_ret_i32(ptr addrspace(1) %out, ptr addrspace(3) %ptr) #0 {
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p3(ptr addrspace(3) %ptr, i32 42, i32 0, i32 0, i1 false)
  store i32 %result, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_inc_ret_i32_offset:
; CIVI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 42
; GCN: ds_inc_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, [[K]] offset:16
define amdgpu_kernel void @lds_atomic_inc_ret_i32_offset(ptr addrspace(1) %out, ptr addrspace(3) %ptr) #0 {
  %gep = getelementptr i32, ptr addrspace(3) %ptr, i32 4
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p3(ptr addrspace(3) %gep, i32 42, i32 0, i32 0, i1 false)
  store i32 %result, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_inc_noret_i32:
; CIVI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: s_load_dword [[SPTR:s[0-9]+]],
; GCN-DAG: v_mov_b32_e32 [[DATA:v[0-9]+]], 4
; GCN-DAG: v_mov_b32_e32 [[VPTR:v[0-9]+]], [[SPTR]]
; GCN: ds_inc_u32 [[VPTR]], [[DATA]]
define amdgpu_kernel void @lds_atomic_inc_noret_i32(ptr addrspace(3) %ptr) nounwind {
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p3(ptr addrspace(3) %ptr, i32 42, i32 0, i32 0, i1 false)
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_inc_noret_i32_offset:
; CIVI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 42
; GCN: ds_inc_u32 v{{[0-9]+}}, [[K]] offset:16
define amdgpu_kernel void @lds_atomic_inc_noret_i32_offset(ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr i32, ptr addrspace(3) %ptr, i32 4
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p3(ptr addrspace(3) %gep, i32 42, i32 0, i32 0, i1 false)
  ret void
}

; GCN-LABEL: {{^}}global_atomic_inc_ret_i32:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 42
; CIVI: buffer_atomic_inc [[K]], off, s{{\[[0-9]+:[0-9]+\]}}, 0 glc{{$}}
; GFX9: global_atomic_inc v{{[0-9]+}}, v{{[0-9]+}}, [[K]], s{{\[[0-9]+:[0-9]+\]}} glc{{$}}
define amdgpu_kernel void @global_atomic_inc_ret_i32(ptr addrspace(1) %out, ptr addrspace(1) %ptr) #0 {
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p1(ptr addrspace(1) %ptr, i32 42, i32 0, i32 0, i1 false)
  store i32 %result, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}global_atomic_inc_ret_i32_offset:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 42
; CIVI: buffer_atomic_inc [[K]], off, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:16 glc{{$}}
; GFX9: global_atomic_inc v{{[0-9]+}}, v{{[0-9]+}}, [[K]], s{{\[[0-9]+:[0-9]+\]}} offset:16 glc{{$}}
define amdgpu_kernel void @global_atomic_inc_ret_i32_offset(ptr addrspace(1) %out, ptr addrspace(1) %ptr) #0 {
  %gep = getelementptr i32, ptr addrspace(1) %ptr, i32 4
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p1(ptr addrspace(1) %gep, i32 42, i32 0, i32 0, i1 false)
  store i32 %result, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}global_atomic_inc_noret_i32:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 42
; CIVI: buffer_atomic_inc [[K]], off, s{{\[[0-9]+:[0-9]+\]}}, 0{{$}}
; GFX9: global_atomic_inc v{{[0-9]+}}, [[K]], s{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @global_atomic_inc_noret_i32(ptr addrspace(1) %ptr) nounwind {
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p1(ptr addrspace(1) %ptr, i32 42, i32 0, i32 0, i1 false)
  ret void
}

; GCN-LABEL: {{^}}global_atomic_inc_noret_i32_offset:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 42
; CIVI: buffer_atomic_inc [[K]], off, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:16{{$}}
; GFX9: global_atomic_inc v{{[0-9]+}}, [[K]], s{{\[[0-9]+:[0-9]+\]}} offset:16{{$}}
define amdgpu_kernel void @global_atomic_inc_noret_i32_offset(ptr addrspace(1) %ptr) nounwind {
  %gep = getelementptr i32, ptr addrspace(1) %ptr, i32 4
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p1(ptr addrspace(1) %gep, i32 42, i32 0, i32 0, i1 false)
  ret void
}

; GCN-LABEL: {{^}}global_atomic_inc_ret_i32_offset_addr64:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 42
; CI: buffer_atomic_inc [[K]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:20 glc{{$}}
; VI: flat_atomic_inc v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, [[K]] glc{{$}}
define amdgpu_kernel void @global_atomic_inc_ret_i32_offset_addr64(ptr addrspace(1) %out, ptr addrspace(1) %ptr) #0 {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.tid = getelementptr i32, ptr addrspace(1) %ptr, i32 %id
  %out.gep = getelementptr i32, ptr addrspace(1) %out, i32 %id
  %gep = getelementptr i32, ptr addrspace(1) %gep.tid, i32 5
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p1(ptr addrspace(1) %gep, i32 42, i32 0, i32 0, i1 false)
  store i32 %result, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}global_atomic_inc_noret_i32_offset_addr64:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 42
; CI: buffer_atomic_inc [[K]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:20{{$}}
; VI: flat_atomic_inc v{{\[[0-9]+:[0-9]+\]}}, [[K]]{{$}}
define amdgpu_kernel void @global_atomic_inc_noret_i32_offset_addr64(ptr addrspace(1) %ptr) #0 {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.tid = getelementptr i32, ptr addrspace(1) %ptr, i32 %id
  %gep = getelementptr i32, ptr addrspace(1) %gep.tid, i32 5
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p1(ptr addrspace(1) %gep, i32 42, i32 0, i32 0, i1 false)
  ret void
}

@lds0 = addrspace(3) global [512 x i32] undef, align 4

; GCN-LABEL: {{^}}atomic_inc_shl_base_lds_0_i32:
; GCN: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 2, {{v[0-9]+}}
; GCN: ds_inc_rtn_u32 {{v[0-9]+}}, [[PTR]], {{v[0-9]+}} offset:8
define amdgpu_kernel void @atomic_inc_shl_base_lds_0_i32(ptr addrspace(1) %out, ptr addrspace(1) %add_use) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x i32], ptr addrspace(3) @lds0, i32 0, i32 %idx.0
  %val0 = call i32 @llvm.amdgcn.atomic.inc.i32.p3(ptr addrspace(3) %arrayidx0, i32 9, i32 0, i32 0, i1 false)
  store i32 %idx.0, ptr addrspace(1) %add_use
  store i32 %val0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_inc_ret_i64:
; GCN-DAG: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; GCN-DAG: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; GCN: ds_inc_rtn_u64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, v[[[KLO]]:[[KHI]]]{{$}}
define amdgpu_kernel void @lds_atomic_inc_ret_i64(ptr addrspace(1) %out, ptr addrspace(3) %ptr) #0 {
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p3(ptr addrspace(3) %ptr, i64 42, i32 0, i32 0, i1 false)
  store i64 %result, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_inc_ret_i64_offset:
; GCN-DAG: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; GCN-DAG: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; GCN: ds_inc_rtn_u64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, v[[[KLO]]:[[KHI]]] offset:32
define amdgpu_kernel void @lds_atomic_inc_ret_i64_offset(ptr addrspace(1) %out, ptr addrspace(3) %ptr) #0 {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i32 4
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p3(ptr addrspace(3) %gep, i64 42, i32 0, i32 0, i1 false)
  store i64 %result, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_inc_noret_i64:
; GCN-DAG: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; GCN-DAG: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; GCN: ds_inc_u64 v{{[0-9]+}}, v[[[KLO]]:[[KHI]]]{{$}}
define amdgpu_kernel void @lds_atomic_inc_noret_i64(ptr addrspace(3) %ptr) nounwind {
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p3(ptr addrspace(3) %ptr, i64 42, i32 0, i32 0, i1 false)
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_inc_noret_i64_offset:
; GCN-DAG: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; GCN-DAG: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; GCN: ds_inc_u64 v{{[0-9]+}}, v[[[KLO]]:[[KHI]]] offset:32{{$}}
define amdgpu_kernel void @lds_atomic_inc_noret_i64_offset(ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i32 4
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p3(ptr addrspace(3) %gep, i64 42, i32 0, i32 0, i1 false)
  ret void
}

; GCN-LABEL: {{^}}global_atomic_inc_ret_i64:
; GCN-DAG: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; GCN-DAG: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; GFX9: v_mov_b32_e32 v[[ZERO:[0-9]+]], 0{{$}}
; CIVI: buffer_atomic_inc_x2 v[[[KLO]]:[[KHI]]], off, s{{\[[0-9]+:[0-9]+\]}}, 0 glc{{$}}
; GFX9: global_atomic_inc_x2 v{{\[[0-9]+:[0-9]+\]}}, v[[ZERO]], v[[[KLO]]:[[KHI]]], s{{\[[0-9]+:[0-9]+\]}} glc{{$}}
define amdgpu_kernel void @global_atomic_inc_ret_i64(ptr addrspace(1) %out, ptr addrspace(1) %ptr) #0 {
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p1(ptr addrspace(1) %ptr, i64 42, i32 0, i32 0, i1 false)
  store i64 %result, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}global_atomic_inc_ret_i64_offset:
; GCN: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; GCN: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; GFX9: v_mov_b32_e32 v[[ZERO:[0-9]+]], 0{{$}}
; CIVI: buffer_atomic_inc_x2 v[[[KLO]]:[[KHI]]], off, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:32 glc{{$}}
; GFX9: global_atomic_inc_x2 v{{\[[0-9]+:[0-9]+\]}}, v[[ZERO]], v[[[KLO]]:[[KHI]]], s{{\[[0-9]+:[0-9]+\]}} offset:32 glc{{$}}
define amdgpu_kernel void @global_atomic_inc_ret_i64_offset(ptr addrspace(1) %out, ptr addrspace(1) %ptr) #0 {
  %gep = getelementptr i64, ptr addrspace(1) %ptr, i32 4
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p1(ptr addrspace(1) %gep, i64 42, i32 0, i32 0, i1 false)
  store i64 %result, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}global_atomic_inc_noret_i64:
; GCN-DAG: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; GCN-DAG: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; GFX9: v_mov_b32_e32 v[[ZERO:[0-9]+]], 0{{$}}
; CIVI: buffer_atomic_inc_x2 v[[[KLO]]:[[KHI]]], off, s{{\[[0-9]+:[0-9]+\]}}, 0{{$}}

; GFX9: global_atomic_inc_x2 v[[ZERO]], v[[[KLO]]:[[KHI]]], s{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @global_atomic_inc_noret_i64(ptr addrspace(1) %ptr) nounwind {
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p1(ptr addrspace(1) %ptr, i64 42, i32 0, i32 0, i1 false)
  ret void
}

; GCN-LABEL: {{^}}global_atomic_inc_noret_i64_offset:
; GCN-DAG: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; GCN-DAG: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; GFX9: v_mov_b32_e32 v[[ZERO:[0-9]+]], 0{{$}}
; CIVI: buffer_atomic_inc_x2 v[[[KLO]]:[[KHI]]], off, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:32{{$}}
; GFX9: global_atomic_inc_x2 v[[ZERO]], v[[[KLO]]:[[KHI]]], s{{\[[0-9]+:[0-9]+\]}} offset:32{{$}}
define amdgpu_kernel void @global_atomic_inc_noret_i64_offset(ptr addrspace(1) %ptr) nounwind {
  %gep = getelementptr i64, ptr addrspace(1) %ptr, i32 4
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p1(ptr addrspace(1) %gep, i64 42, i32 0, i32 0, i1 false)
  ret void
}

; GCN-LABEL: {{^}}global_atomic_inc_ret_i64_offset_addr64:
; GCN: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; CI: v_mov_b32_e32 v{{[0-9]+}}, 0{{$}}
; GCN: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; CI: buffer_atomic_inc_x2 v[[[KLO]]:[[KHI]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:40 glc{{$}}
; VI: flat_atomic_inc_x2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v[[[KLO]]:[[KHI]]] glc{{$}}
define amdgpu_kernel void @global_atomic_inc_ret_i64_offset_addr64(ptr addrspace(1) %out, ptr addrspace(1) %ptr) #0 {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.tid = getelementptr i64, ptr addrspace(1) %ptr, i32 %id
  %out.gep = getelementptr i64, ptr addrspace(1) %out, i32 %id
  %gep = getelementptr i64, ptr addrspace(1) %gep.tid, i32 5
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p1(ptr addrspace(1) %gep, i64 42, i32 0, i32 0, i1 false)
  store i64 %result, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}global_atomic_inc_noret_i64_offset_addr64:
; GCN: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; CI: v_mov_b32_e32 v{{[0-9]+}}, 0{{$}}
; GCN: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; CI: buffer_atomic_inc_x2 v[[[KLO]]:[[KHI]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:40{{$}}
; VI: flat_atomic_inc_x2 v{{\[[0-9]+:[0-9]+\]}}, v[[[KLO]]:[[KHI]]]{{$}}
define amdgpu_kernel void @global_atomic_inc_noret_i64_offset_addr64(ptr addrspace(1) %ptr) #0 {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.tid = getelementptr i64, ptr addrspace(1) %ptr, i32 %id
  %gep = getelementptr i64, ptr addrspace(1) %gep.tid, i32 5
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p1(ptr addrspace(1) %gep, i64 42, i32 0, i32 0, i1 false)
  ret void
}

; GCN-LABEL: {{^}}flat_atomic_inc_ret_i32:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 42
; GCN: flat_atomic_inc v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, [[K]] glc{{$}}
define amdgpu_kernel void @flat_atomic_inc_ret_i32(ptr %out, ptr %ptr) #0 {
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p0(ptr %ptr, i32 42, i32 0, i32 0, i1 false)
  store i32 %result, ptr %out
  ret void
}

; GCN-LABEL: {{^}}flat_atomic_inc_ret_i32_offset:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 42
; CIVI: flat_atomic_inc v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, [[K]] glc{{$}}
; GFX9: flat_atomic_inc v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, [[K]] offset:16 glc{{$}}
define amdgpu_kernel void @flat_atomic_inc_ret_i32_offset(ptr %out, ptr %ptr) #0 {
  %gep = getelementptr i32, ptr %ptr, i32 4
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p0(ptr %gep, i32 42, i32 0, i32 0, i1 false)
  store i32 %result, ptr %out
  ret void
}

; GCN-LABEL: {{^}}flat_atomic_inc_noret_i32:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 42
; GCN: flat_atomic_inc v{{\[[0-9]+:[0-9]+\]}}, [[K]]{{$}}
define amdgpu_kernel void @flat_atomic_inc_noret_i32(ptr %ptr) nounwind {
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p0(ptr %ptr, i32 42, i32 0, i32 0, i1 false)
  ret void
}

; GCN-LABEL: {{^}}flat_atomic_inc_noret_i32_offset:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 42
; CIVI: flat_atomic_inc v{{\[[0-9]+:[0-9]+\]}}, [[K]]{{$}}
; GFX9: flat_atomic_inc v{{\[[0-9]+:[0-9]+\]}}, [[K]] offset:16{{$}}
define amdgpu_kernel void @flat_atomic_inc_noret_i32_offset(ptr %ptr) nounwind {
  %gep = getelementptr i32, ptr %ptr, i32 4
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p0(ptr %gep, i32 42, i32 0, i32 0, i1 false)
  ret void
}

; GCN-LABEL: {{^}}flat_atomic_inc_ret_i32_offset_addr64:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 42
; CIVI: flat_atomic_inc v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, [[K]] glc{{$}}
; GFX9: flat_atomic_inc v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, [[K]] offset:20 glc{{$}}
define amdgpu_kernel void @flat_atomic_inc_ret_i32_offset_addr64(ptr %out, ptr %ptr) #0 {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.tid = getelementptr i32, ptr %ptr, i32 %id
  %out.gep = getelementptr i32, ptr %out, i32 %id
  %gep = getelementptr i32, ptr %gep.tid, i32 5
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p0(ptr %gep, i32 42, i32 0, i32 0, i1 false)
  store i32 %result, ptr %out.gep
  ret void
}

; GCN-LABEL: {{^}}flat_atomic_inc_noret_i32_offset_addr64:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 42
; CIVI: flat_atomic_inc v{{\[[0-9]+:[0-9]+\]}}, [[K]]{{$}}
; GFX9: flat_atomic_inc v{{\[[0-9]+:[0-9]+\]}}, [[K]] offset:20{{$}}
define amdgpu_kernel void @flat_atomic_inc_noret_i32_offset_addr64(ptr %ptr) #0 {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.tid = getelementptr i32, ptr %ptr, i32 %id
  %gep = getelementptr i32, ptr %gep.tid, i32 5
  %result = call i32 @llvm.amdgcn.atomic.inc.i32.p0(ptr %gep, i32 42, i32 0, i32 0, i1 false)
  ret void
}

@lds1 = addrspace(3) global [512 x i64] undef, align 8

; GCN-LABEL: {{^}}atomic_inc_shl_base_lds_0_i64:
; GCN: v_lshlrev_b32_e32 [[PTR:v[0-9]+]], 3, {{v[0-9]+}}
; GCN: ds_inc_rtn_u64 v{{\[[0-9]+:[0-9]+\]}}, [[PTR]], v{{\[[0-9]+:[0-9]+\]}} offset:16
define amdgpu_kernel void @atomic_inc_shl_base_lds_0_i64(ptr addrspace(1) %out, ptr addrspace(1) %add_use) #0 {
  %tid.x = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %idx.0 = add nsw i32 %tid.x, 2
  %arrayidx0 = getelementptr inbounds [512 x i64], ptr addrspace(3) @lds1, i32 0, i32 %idx.0
  %val0 = call i64 @llvm.amdgcn.atomic.inc.i64.p3(ptr addrspace(3) %arrayidx0, i64 9, i32 0, i32 0, i1 false)
  store i32 %idx.0, ptr addrspace(1) %add_use
  store i64 %val0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}flat_atomic_inc_ret_i64:
; GCN-DAG: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; GCN-DAG: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; GCN: flat_atomic_inc_x2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v[[[KLO]]:[[KHI]]] glc{{$}}
define amdgpu_kernel void @flat_atomic_inc_ret_i64(ptr %out, ptr %ptr) #0 {
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p0(ptr %ptr, i64 42, i32 0, i32 0, i1 false)
  store i64 %result, ptr %out
  ret void
}

; GCN-LABEL: {{^}}flat_atomic_inc_ret_i64_offset:
; GCN-DAG: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; GCN-DAG: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; CIVI: flat_atomic_inc_x2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v[[[KLO]]:[[KHI]]] glc{{$}}
; GFX9: flat_atomic_inc_x2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v[[[KLO]]:[[KHI]]] offset:32 glc{{$}}
define amdgpu_kernel void @flat_atomic_inc_ret_i64_offset(ptr %out, ptr %ptr) #0 {
  %gep = getelementptr i64, ptr %ptr, i32 4
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p0(ptr %gep, i64 42, i32 0, i32 0, i1 false)
  store i64 %result, ptr %out
  ret void
}

; GCN-LABEL: {{^}}flat_atomic_inc_noret_i64:
; GCN-DAG: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; GCN-DAG: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; GCN: flat_atomic_inc_x2 v{{\[[0-9]+:[0-9]+\]}}, v[[[KLO]]:[[KHI]]{{\]$}}
define amdgpu_kernel void @flat_atomic_inc_noret_i64(ptr %ptr) nounwind {
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p0(ptr %ptr, i64 42, i32 0, i32 0, i1 false)
  ret void
}

; GCN-LABEL: {{^}}flat_atomic_inc_noret_i64_offset:
; GCN-DAG: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; GCN-DAG: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; CIVI: flat_atomic_inc_x2 v{{\[[0-9]+:[0-9]+\]}}, v[[[KLO]]:[[KHI]]{{\]$}}
; GFX9: flat_atomic_inc_x2 v{{\[[0-9]+:[0-9]+\]}}, v[[[KLO]]:[[KHI]]] offset:32{{$}}
define amdgpu_kernel void @flat_atomic_inc_noret_i64_offset(ptr %ptr) nounwind {
  %gep = getelementptr i64, ptr %ptr, i32 4
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p0(ptr %gep, i64 42, i32 0, i32 0, i1 false)
  ret void
}

; GCN-LABEL: {{^}}flat_atomic_inc_ret_i64_offset_addr64:
; GCN: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; GCN: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; CIVI: flat_atomic_inc_x2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v[[[KLO]]:[[KHI]]] glc{{$}}
; GFX9: flat_atomic_inc_x2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v[[[KLO]]:[[KHI]]] offset:40 glc{{$}}
define amdgpu_kernel void @flat_atomic_inc_ret_i64_offset_addr64(ptr %out, ptr %ptr) #0 {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.tid = getelementptr i64, ptr %ptr, i32 %id
  %out.gep = getelementptr i64, ptr %out, i32 %id
  %gep = getelementptr i64, ptr %gep.tid, i32 5
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p0(ptr %gep, i64 42, i32 0, i32 0, i1 false)
  store i64 %result, ptr %out.gep
  ret void
}

; GCN-LABEL: {{^}}flat_atomic_inc_noret_i64_offset_addr64:
; GCN: v_mov_b32_e32 v[[KLO:[0-9]+]], 42
; GCN: v_mov_b32_e32 v[[KHI:[0-9]+]], 0{{$}}
; CIVI: flat_atomic_inc_x2 v{{\[[0-9]+:[0-9]+\]}}, v[[[KLO]]:[[KHI]]{{\]$}}
; GFX9: flat_atomic_inc_x2 v{{\[[0-9]+:[0-9]+\]}}, v[[[KLO]]:[[KHI]]] offset:40{{$}}
define amdgpu_kernel void @flat_atomic_inc_noret_i64_offset_addr64(ptr %ptr) #0 {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.tid = getelementptr i64, ptr %ptr, i32 %id
  %gep = getelementptr i64, ptr %gep.tid, i32 5
  %result = call i64 @llvm.amdgcn.atomic.inc.i64.p0(ptr %gep, i64 42, i32 0, i32 0, i1 false)
  ret void
}

; GCN-LABEL: {{^}}nocse_lds_atomic_inc_ret_i32:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 42
; GCN: ds_inc_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, [[K]]
; GCN: ds_inc_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, [[K]]
define amdgpu_kernel void @nocse_lds_atomic_inc_ret_i32(ptr addrspace(1) %out0, ptr addrspace(1) %out1, ptr addrspace(3) %ptr) #0 {
  %result0 = call i32 @llvm.amdgcn.atomic.inc.i32.p3(ptr addrspace(3) %ptr, i32 42, i32 0, i32 0, i1 false)
  %result1 = call i32 @llvm.amdgcn.atomic.inc.i32.p3(ptr addrspace(3) %ptr, i32 42, i32 0, i32 0, i1 false)

  store i32 %result0, ptr addrspace(1) %out0
  store i32 %result1, ptr addrspace(1) %out1
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind argmemonly }
