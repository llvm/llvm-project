; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

declare i16 @llvm.convert.to.fp16.f32(float) nounwind readnone

; SI-LABEL: {{^}}test_convert_fp32_to_fp16:
; SI: buffer_load_dword [[VAL:v[0-9]+]]
; SI: v_cvt_f16_f32_e32 [[RESULT:v[0-9]+]], [[VAL]]
; SI: buffer_store_short [[RESULT]]
define void @test_convert_fp32_to_fp16(i16 addrspace(1)* noalias %out, float addrspace(1)* noalias %in) nounwind {
  %val = load float addrspace(1)* %in, align 4
  %cvt = call i16 @llvm.convert.to.fp16.f32(float %val) nounwind readnone
  store i16 %cvt, i16 addrspace(1)* %out, align 2
  ret void
}
