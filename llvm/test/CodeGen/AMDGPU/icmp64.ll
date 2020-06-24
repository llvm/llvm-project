; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

; SI-LABEL: {{^}}test_i64_eq:
; SI: v_cmp_eq_u64
define amdgpu_kernel void @test_i64_eq(i32 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %cmp = icmp eq i64 %a, %b
  %result = sext i1 %cmp to i32
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_i64_ne:
; SI: v_cmp_ne_u64
define amdgpu_kernel void @test_i64_ne(i32 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %cmp = icmp ne i64 %a, %b
  %result = sext i1 %cmp to i32
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_i64_slt:
; SI: v_cmp_lt_i64
define amdgpu_kernel void @test_i64_slt(i32 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %cmp = icmp slt i64 %a, %b
  %result = sext i1 %cmp to i32
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_i64_ult:
; SI: v_cmp_lt_u64
define amdgpu_kernel void @test_i64_ult(i32 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %cmp = icmp ult i64 %a, %b
  %result = sext i1 %cmp to i32
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_i64_sle:
; SI: v_cmp_le_i64
define amdgpu_kernel void @test_i64_sle(i32 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %cmp = icmp sle i64 %a, %b
  %result = sext i1 %cmp to i32
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_i64_ule:
; SI: v_cmp_le_u64
define amdgpu_kernel void @test_i64_ule(i32 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %cmp = icmp ule i64 %a, %b
  %result = sext i1 %cmp to i32
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_i64_sgt:
; SI: v_cmp_gt_i64
define amdgpu_kernel void @test_i64_sgt(i32 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %cmp = icmp sgt i64 %a, %b
  %result = sext i1 %cmp to i32
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_i64_ugt:
; SI: v_cmp_gt_u64
define amdgpu_kernel void @test_i64_ugt(i32 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %cmp = icmp ugt i64 %a, %b
  %result = sext i1 %cmp to i32
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_i64_sge:
; SI: v_cmp_ge_i64
define amdgpu_kernel void @test_i64_sge(i32 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %cmp = icmp sge i64 %a, %b
  %result = sext i1 %cmp to i32
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_i64_uge:
; SI: v_cmp_ge_u64
define amdgpu_kernel void @test_i64_uge(i32 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %cmp = icmp uge i64 %a, %b
  %result = sext i1 %cmp to i32
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

