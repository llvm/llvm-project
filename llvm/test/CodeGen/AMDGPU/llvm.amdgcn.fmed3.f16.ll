; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}test_fmed3_f16:
; GCN: v_med3_f16 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @test_fmed3_f16(ptr addrspace(1) %out, i32 %src0.arg, i32 %src1.arg, i32 %src2.arg) #1 {
  %src0.f16 = trunc i32 %src0.arg to i16
  %src0 = bitcast i16 %src0.f16 to half
  %src1.f16 = trunc i32 %src1.arg to i16
  %src1 = bitcast i16 %src1.f16 to half
  %src2.f16 = trunc i32 %src2.arg to i16
  %src2 = bitcast i16 %src2.f16 to half
  %mad = call half @llvm.amdgcn.fmed3.f16(half %src0, half %src1, half %src2)
  store half %mad, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}test_fmed3_srcmods_f16:
; GCN: v_med3_f16 v{{[0-9]+}}, -s{{[0-9]+}}, |v{{[0-9]+}}|, -|v{{[0-9]+}}|
define amdgpu_kernel void @test_fmed3_srcmods_f16(ptr addrspace(1) %out, i32 %src0.arg, i32 %src1.arg, i32 %src2.arg) #1 {
  %src0.f16 = trunc i32 %src0.arg to i16
  %src0 = bitcast i16 %src0.f16 to half
  %src1.f16 = trunc i32 %src1.arg to i16
  %src1 = bitcast i16 %src1.f16 to half
  %src2.f16 = trunc i32 %src2.arg to i16
  %src2 = bitcast i16 %src2.f16 to half
  %src0.fneg = fsub half -0.0, %src0
  %src1.fabs = call half @llvm.fabs.f16(half %src1)
  %src2.fabs = call half @llvm.fabs.f16(half %src2)
  %src2.fneg.fabs = fsub half -0.0, %src2.fabs
  %mad = call half @llvm.amdgcn.fmed3.f16(half %src0.fneg, half %src1.fabs, half %src2.fneg.fabs)
  store half %mad, ptr addrspace(1) %out
  ret void
}

declare half @llvm.amdgcn.fmed3.f16(half, half, half) #0
declare half @llvm.fabs.f16(half) #0

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
