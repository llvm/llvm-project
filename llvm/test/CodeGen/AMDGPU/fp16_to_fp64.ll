; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=FUNC %s

declare double @llvm.convert.from.fp16.f64(i16) nounwind readnone

; FUNC-LABEL: {{^}}test_convert_fp16_to_fp64:
; GCN: buffer_load_ushort [[VAL:v[0-9]+]]
; GCN: v_cvt_f32_f16_e32 [[RESULT32:v[0-9]+]], [[VAL]]
; GCN: v_cvt_f64_f32_e32 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[RESULT32]]
; GCN: buffer_store_dwordx2 [[RESULT]]
define amdgpu_kernel void @test_convert_fp16_to_fp64(ptr addrspace(1) noalias %out, ptr addrspace(1) noalias %in) nounwind {
  %val = load i16, ptr addrspace(1) %in, align 2
  %cvt = call double @llvm.convert.from.fp16.f64(i16 %val) nounwind readnone
  store double %cvt, ptr addrspace(1) %out, align 4
  ret void
}
