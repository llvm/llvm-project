; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,VI %s

; GCN-LABEL: {{^}}test_i64_eq:
; VI: s_cmp_eq_u64
; SI: v_cmp_eq_u64
define amdgpu_kernel void @test_i64_eq(ptr addrspace(1) %out, i64 %a, i64 %b) nounwind {
  %cmp = icmp eq i64 %a, %b
  %result = sext i1 %cmp to i32
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_i64_ne:
; VI: s_cmp_lg_u64
; SI: v_cmp_ne_u64
define amdgpu_kernel void @test_i64_ne(ptr addrspace(1) %out, i64 %a, i64 %b) nounwind {
  %cmp = icmp ne i64 %a, %b
  %result = sext i1 %cmp to i32
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_i64_slt:
; GCN: v_cmp_lt_i64
define amdgpu_kernel void @test_i64_slt(ptr addrspace(1) %out, i64 %a, i64 %b) nounwind {
  %cmp = icmp slt i64 %a, %b
  %result = sext i1 %cmp to i32
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_i64_ult:
; GCN: v_cmp_lt_u64
define amdgpu_kernel void @test_i64_ult(ptr addrspace(1) %out, i64 %a, i64 %b) nounwind {
  %cmp = icmp ult i64 %a, %b
  %result = sext i1 %cmp to i32
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_i64_sle:
; GCN: v_cmp_le_i64
define amdgpu_kernel void @test_i64_sle(ptr addrspace(1) %out, i64 %a, i64 %b) nounwind {
  %cmp = icmp sle i64 %a, %b
  %result = sext i1 %cmp to i32
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_i64_ule:
; GCN: v_cmp_le_u64
define amdgpu_kernel void @test_i64_ule(ptr addrspace(1) %out, i64 %a, i64 %b) nounwind {
  %cmp = icmp ule i64 %a, %b
  %result = sext i1 %cmp to i32
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_i64_sgt:
; GCN: v_cmp_gt_i64
define amdgpu_kernel void @test_i64_sgt(ptr addrspace(1) %out, i64 %a, i64 %b) nounwind {
  %cmp = icmp sgt i64 %a, %b
  %result = sext i1 %cmp to i32
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_i64_ugt:
; GCN: v_cmp_gt_u64
define amdgpu_kernel void @test_i64_ugt(ptr addrspace(1) %out, i64 %a, i64 %b) nounwind {
  %cmp = icmp ugt i64 %a, %b
  %result = sext i1 %cmp to i32
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_i64_sge:
; GCN: v_cmp_ge_i64
define amdgpu_kernel void @test_i64_sge(ptr addrspace(1) %out, i64 %a, i64 %b) nounwind {
  %cmp = icmp sge i64 %a, %b
  %result = sext i1 %cmp to i32
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_i64_uge:
; GCN: v_cmp_ge_u64
define amdgpu_kernel void @test_i64_uge(ptr addrspace(1) %out, i64 %a, i64 %b) nounwind {
  %cmp = icmp uge i64 %a, %b
  %result = sext i1 %cmp to i32
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

