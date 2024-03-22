; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}test_mul_i24:
; GCN: v_mul_i32_i24
define amdgpu_kernel void @test_mul_i24(ptr addrspace(1) %out, i32 %src1, i32 %src2) nounwind {
  %val = call i32 @llvm.amdgcn.mul.i24(i32 %src1, i32 %src2) nounwind readnone speculatable
  store i32 %val, ptr addrspace(1) %out
  ret void
}

declare i32 @llvm.amdgcn.mul.i24(i32, i32) nounwind readnone speculatable
