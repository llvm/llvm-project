; RUN: llc -mtriple=amdgcn -mcpu=gfx900 < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}test_umed3_i16:
; GCN: v_med3_u16 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @test_umed3_i16(ptr addrspace(1) %out, i32 %src0.arg, i32 %src1.arg, i32 %src2.arg) #1 {
  %src0.i16 = trunc i32 %src0.arg to i16
  %src1.i16 = trunc i32 %src1.arg to i16
  %src2.i16 = trunc i32 %src2.arg to i16
  %med3 = call i16 @llvm.amdgcn.umed3.i16(i16 %src0.i16, i16 %src1.i16, i16 %src2.i16)
  store i16 %med3, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}test_umed3_zero_i16:
; GCN: v_med3_u16 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, 0
define amdgpu_kernel void @test_umed3_zero_i16(ptr addrspace(1) %out, i32 %src0.arg, i32 %src1.arg) #1 {
  %src0.i16 = trunc i32 %src0.arg to i16
  %src1.i16 = trunc i32 %src1.arg to i16
  %med3 = call i16 @llvm.amdgcn.umed3.i16(i16 %src0.i16, i16 %src1.i16, i16 0)
  store i16 %med3, ptr addrspace(1) %out
  ret void
}

declare i16 @llvm.amdgcn.umed3.i16(i16, i16, i16) #0

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
