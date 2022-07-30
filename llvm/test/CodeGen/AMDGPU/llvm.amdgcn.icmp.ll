; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SI %s
; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI %s

declare i64 @llvm.amdgcn.icmp.i32(i32, i32, i32) #0
declare i64 @llvm.amdgcn.icmp.i64(i64, i64, i32) #0
declare i64 @llvm.amdgcn.icmp.i16(i16, i16, i32) #0
declare i64 @llvm.amdgcn.icmp.i1(i1, i1, i32) #0

; GCN-LABEL: {{^}}v_icmp_i32_eq:
; GCN: v_cmp_eq_u32_e64
define amdgpu_kernel void @v_icmp_i32_eq(i64 addrspace(1)* %out, i32 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i32(i32 %src, i32 100, i32 32)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_i32:
; GCN-NOT: v_cmp_eq_u32_e64
define amdgpu_kernel void @v_icmp_i32(i64 addrspace(1)* %out, i32 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i32(i32 %src, i32 100, i32 30)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_i32_ne:
; GCN: v_cmp_ne_u32_e64
define amdgpu_kernel void @v_icmp_i32_ne(i64 addrspace(1)* %out, i32 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i32(i32 %src, i32 100, i32 33)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_i32_ugt:
; GCN: v_cmp_gt_u32_e64
define amdgpu_kernel void @v_icmp_i32_ugt(i64 addrspace(1)* %out, i32 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i32(i32 %src, i32 100, i32 34)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_i32_uge:
; GCN: v_cmp_ge_u32_e64
define amdgpu_kernel void @v_icmp_i32_uge(i64 addrspace(1)* %out, i32 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i32(i32 %src, i32 100, i32 35)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_i32_ult:
; GCN: v_cmp_lt_u32_e64
define amdgpu_kernel void @v_icmp_i32_ult(i64 addrspace(1)* %out, i32 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i32(i32 %src, i32 100, i32 36)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_i32_ule:
; GCN: v_cmp_le_u32_e64
define amdgpu_kernel void @v_icmp_i32_ule(i64 addrspace(1)* %out, i32 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i32(i32 %src, i32 100, i32 37)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_i32_sgt:
; GCN: v_cmp_gt_i32_e64
define amdgpu_kernel void @v_icmp_i32_sgt(i64 addrspace(1)* %out, i32 %src) #1 {
  %result = call i64 @llvm.amdgcn.icmp.i32(i32 %src, i32 100, i32 38)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_i32_sge:
; GCN: v_cmp_ge_i32_e64
define amdgpu_kernel void @v_icmp_i32_sge(i64 addrspace(1)* %out, i32 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i32(i32 %src, i32 100, i32 39)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_i32_slt:
; GCN: v_cmp_lt_i32_e64
define amdgpu_kernel void @v_icmp_i32_slt(i64 addrspace(1)* %out, i32 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i32(i32 %src, i32 100, i32 40)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}
; GCN-LABEL: {{^}}v_icmp_i32_sle:
; GCN: v_cmp_le_i32_e64
define amdgpu_kernel void @v_icmp_i32_sle(i64 addrspace(1)* %out, i32 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i32(i32 %src, i32 100, i32 41)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_i64_eq:
; GCN: v_cmp_eq_u64_e64
define amdgpu_kernel void @v_icmp_i64_eq(i64 addrspace(1)* %out, i64 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i64(i64 %src, i64 100, i32 32)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_i64_ne:
; GCN: v_cmp_ne_u64_e64
define amdgpu_kernel void @v_icmp_i64_ne(i64 addrspace(1)* %out, i64 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i64(i64 %src, i64 100, i32 33)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_u64_ugt:
; GCN: v_cmp_gt_u64_e64
define amdgpu_kernel void @v_icmp_u64_ugt(i64 addrspace(1)* %out, i64 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i64(i64 %src, i64 100, i32 34)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_u64_uge:
; GCN: v_cmp_ge_u64_e64
define amdgpu_kernel void @v_icmp_u64_uge(i64 addrspace(1)* %out, i64 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i64(i64 %src, i64 100, i32 35)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_u64_ult:
; GCN: v_cmp_lt_u64_e64
define amdgpu_kernel void @v_icmp_u64_ult(i64 addrspace(1)* %out, i64 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i64(i64 %src, i64 100, i32 36)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_u64_ule:
; GCN: v_cmp_le_u64_e64
define amdgpu_kernel void @v_icmp_u64_ule(i64 addrspace(1)* %out, i64 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i64(i64 %src, i64 100, i32 37)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_i64_sgt:
; GCN: v_cmp_gt_i64_e64
define amdgpu_kernel void @v_icmp_i64_sgt(i64 addrspace(1)* %out, i64 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i64(i64 %src, i64 100, i32 38)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_i64_sge:
; GCN: v_cmp_ge_i64_e64
define amdgpu_kernel void @v_icmp_i64_sge(i64 addrspace(1)* %out, i64 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i64(i64 %src, i64 100, i32 39)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_i64_slt:
; GCN: v_cmp_lt_i64_e64
define amdgpu_kernel void @v_icmp_i64_slt(i64 addrspace(1)* %out, i64 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i64(i64 %src, i64 100, i32 40)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}
; GCN-LABEL: {{^}}v_icmp_i64_sle:
; GCN: v_cmp_le_i64_e64
define amdgpu_kernel void @v_icmp_i64_sle(i64 addrspace(1)* %out, i64 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i64(i64 %src, i64 100, i32 41)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; VI: v_cmp_eq_u16_e64

; SI-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0x64
; SI-DAG: s_and_b32 [[CVT:s[0-9]+]], s{{[0-9]+}}, 0xffff{{$}}
; SI: v_cmp_eq_u32_e64 s{{\[[0-9]+:[0-9]+\]}}, [[CVT]], [[K]]
define amdgpu_kernel void @v_icmp_i16_eq(i64 addrspace(1)* %out, i16 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i16(i16 %src, i16 100, i32 32)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_i16:
; GCN-NOT: v_cmp_eq_
define amdgpu_kernel void @v_icmp_i16(i64 addrspace(1)* %out, i16 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i16(i16 %src, i16 100, i32 30)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}
; GCN-LABEL: {{^}}v_icmp_i16_ne:
; VI: v_cmp_ne_u16_e64

; SI-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0x64
; SI-DAG: s_and_b32 [[CVT:s[0-9]+]], s{{[0-9]+}}, 0xffff{{$}}
; SI: v_cmp_ne_u32_e64 s{{\[[0-9]+:[0-9]+\]}}, [[CVT]], [[K]]
define amdgpu_kernel void @v_icmp_i16_ne(i64 addrspace(1)* %out, i16 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i16(i16 %src, i16 100, i32 33)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_i16_ugt:
; VI: v_cmp_gt_u16_e64

; SI-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0x64
; SI-DAG: s_and_b32 [[CVT:s[0-9]+]], s{{[0-9]+}}, 0xffff{{$}}
; SI: v_cmp_gt_u32_e64 s{{\[[0-9]+:[0-9]+\]}}, [[CVT]], [[K]]
define amdgpu_kernel void @v_icmp_i16_ugt(i64 addrspace(1)* %out, i16 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i16(i16 %src, i16 100, i32 34)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_i16_uge:
; VI: v_cmp_ge_u16_e64

; SI-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0x64
; SI-DAG: s_and_b32 [[CVT:s[0-9]+]], s{{[0-9]+}}, 0xffff{{$}}
; SI: v_cmp_ge_u32_e64 s{{\[[0-9]+:[0-9]+\]}}, [[CVT]], [[K]]
define amdgpu_kernel void @v_icmp_i16_uge(i64 addrspace(1)* %out, i16 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i16(i16 %src, i16 100, i32 35)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_i16_ult:
; VI: v_cmp_lt_u16_e64

; SI-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0x64
; SI-DAG: s_and_b32 [[CVT:s[0-9]+]], s{{[0-9]+}}, 0xffff{{$}}
; SI: v_cmp_lt_u32_e64 s{{\[[0-9]+:[0-9]+\]}}, [[CVT]], [[K]]
define amdgpu_kernel void @v_icmp_i16_ult(i64 addrspace(1)* %out, i16 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i16(i16 %src, i16 100, i32 36)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_i16_ule:
; VI: v_cmp_le_u16_e64

; SI-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0x64
; SI-DAG: s_and_b32 [[CVT:s[0-9]+]], s{{[0-9]+}}, 0xffff{{$}}
; SI: v_cmp_le_u32_e64 s{{\[[0-9]+:[0-9]+\]}}, [[CVT]], [[K]]
define amdgpu_kernel void @v_icmp_i16_ule(i64 addrspace(1)* %out, i16 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i16(i16 %src, i16 100, i32 37)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_i16_sgt:
; VI: v_cmp_gt_i16_e64

; SI-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0x64
; SI-DAG: s_sext_i32_i16 [[CVT:s[0-9]+]], s{{[0-9]+}}
; SI: v_cmp_gt_i32_e64 s{{\[[0-9]+:[0-9]+\]}}, [[CVT]], [[K]]
define amdgpu_kernel void @v_icmp_i16_sgt(i64 addrspace(1)* %out, i16 %src) #1 {
  %result = call i64 @llvm.amdgcn.icmp.i16(i16 %src, i16 100, i32 38)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_i16_sge:
; VI: v_cmp_ge_i16_e64

; SI-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0x64
; SI-DAG: s_sext_i32_i16 [[CVT:s[0-9]+]], s{{[0-9]+}}
; SI: v_cmp_ge_i32_e64 s{{\[[0-9]+:[0-9]+\]}}, [[CVT]], [[K]]
define amdgpu_kernel void @v_icmp_i16_sge(i64 addrspace(1)* %out, i16 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i16(i16 %src, i16 100, i32 39)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_i16_slt:
; VI: v_cmp_lt_i16_e64

; SI-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0x64
; SI-DAG: s_sext_i32_i16 [[CVT:s[0-9]+]], s{{[0-9]+}}
; SI: v_cmp_lt_i32_e64 s{{\[[0-9]+:[0-9]+\]}}, [[CVT]], [[K]]
define amdgpu_kernel void @v_icmp_i16_slt(i64 addrspace(1)* %out, i16 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i16(i16 %src, i16 100, i32 40)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}
; GCN-LABEL: {{^}}v_icmp_i16_sle:
; VI: v_cmp_le_i16_e64

; SI-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0x64
; SI-DAG: s_sext_i32_i16 [[CVT:s[0-9]+]], s{{[0-9]+}}
; SI: v_cmp_le_i32_e64 s{{\[[0-9]+:[0-9]+\]}}, [[CVT]], [[K]]
define amdgpu_kernel void @v_icmp_i16_sle(i64 addrspace(1)* %out, i16 %src) {
  %result = call i64 @llvm.amdgcn.icmp.i16(i16 %src, i16 100, i32 41)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_icmp_i1_ne0:
; GCN: s_cmp_gt_u32
; GCN: s_cselect_b64 s[[C0:\[[0-9]+:[0-9]+\]]], -1, 0
; GCN: s_cmp_gt_u32
; GCN: s_cselect_b64 s[[C1:\[[0-9]+:[0-9]+\]]], -1, 0
; GCN: s_and_b64 s[[SRC:\[[0-9]+:[0-9]+\]]], s[[C0]], s[[C1]]
; SI-NEXT: s_mov_b32 s{{[0-9]+}}, -1
; SI-NEXT: s_mov_b32
; SI-NEXT: s_mov_b32
; GCN-NEXT: v_mov_b32_e32
; GCN-NEXT: v_mov_b32_e32
; GCN: {{global|flat|buffer}}_store_dwordx2
define amdgpu_kernel void @v_icmp_i1_ne0(i64 addrspace(1)* %out, i32 %a, i32 %b) {
  %c0 = icmp ugt i32 %a, 1
  %c1 = icmp ugt i32 %b, 2
  %src = and i1 %c0, %c1
  %result = call i64 @llvm.amdgcn.icmp.i1(i1 %src, i1 false, i32 33)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind readnone convergent }
