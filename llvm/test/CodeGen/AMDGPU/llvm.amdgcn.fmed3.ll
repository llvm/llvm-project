; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}test_fmed3:
; GCN: v_med3_f32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define void @test_fmed3(float addrspace(1)* %out, float %src0, float %src1, float %src2) #1 {
  %mad = call float @llvm.amdgcn.fmed3.f32(float %src0, float %src1, float %src2)
  store float %mad, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fmed3_srcmods:
; GCN: v_med3_f32 v{{[0-9]+}}, -s{{[0-9]+}}, |v{{[0-9]+}}|, -|v{{[0-9]+}}|
define void @test_fmed3_srcmods(float addrspace(1)* %out, float %src0, float %src1, float %src2) #1 {
  %src0.fneg = fsub float -0.0, %src0
  %src1.fabs = call float @llvm.fabs.f32(float %src1)
  %src2.fabs = call float @llvm.fabs.f32(float %src2)
  %src2.fneg.fabs = fsub float -0.0, %src2.fabs
  %mad = call float @llvm.amdgcn.fmed3.f32(float %src0.fneg, float %src1.fabs, float %src2.fneg.fabs)
  store float %mad, float addrspace(1)* %out
  ret void
}

declare float @llvm.amdgcn.fmed3.f32(float, float, float) #0
declare float @llvm.fabs.f32(float) #0

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
