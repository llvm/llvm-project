; RUN: llc -mtriple=amdgcn < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}test_smed3:
; GCN: v_med3_i32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @test_smed3(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) #1 {
  %med3 = call i32 @llvm.amdgcn.smed3.i32(i32 %src0, i32 %src1, i32 %src2)
  store i32 %med3, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}test_smed3_multi_use:
; GCN: v_med3_i32 [[MED3:v[0-9]+]], s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_mul_lo_i32 v{{[0-9]+}}, [[MED3]], s{{[0-9]+}}
define amdgpu_kernel void @test_smed3_multi_use(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2, i32 %mul.arg) #1 {
  %med3 = call i32 @llvm.amdgcn.smed3.i32(i32 %src0, i32 %src1, i32 %src2)
  %med3.user = mul i32 %med3, %mul.arg
  store volatile i32 %med3.user, ptr addrspace(1) %out
  store volatile i32 %med3, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}test_smed3_constants:
; GCN: v_med3_i32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, 42
define amdgpu_kernel void @test_smed3_constants(ptr addrspace(1) %out, i32 %src0, i32 %src1) #1 {
  %med3 = call i32 @llvm.amdgcn.smed3.i32(i32 %src0, i32 %src1, i32 42)
  store i32 %med3, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}test_smed3_zero:
; GCN: v_med3_i32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, 0
define amdgpu_kernel void @test_smed3_zero(ptr addrspace(1) %out, i32 %src0, i32 %src1) #1 {
  %med3 = call i32 @llvm.amdgcn.smed3.i32(i32 %src0, i32 %src1, i32 0)
  store i32 %med3, ptr addrspace(1) %out
  ret void
}

declare i32 @llvm.amdgcn.smed3.i32(i32, i32, i32) #0

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
