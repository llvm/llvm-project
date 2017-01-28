; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

declare i16 @llvm.convert.to.fp16.f32(float) nounwind readnone

; FUNC-LABEL: {{^}}test_convert_fp32_to_fp16:
; GCN: buffer_load_dword [[VAL:v[0-9]+]]
; GCN: v_cvt_f16_f32_e32 [[RESULT:v[0-9]+]], [[VAL]]
; GCN: buffer_store_short [[RESULT]]

; EG: MEM_RAT MSKOR
; EG: VTX_READ_32
; EG: FLT32_TO_FLT16
define void @test_convert_fp32_to_fp16(i16 addrspace(1)* noalias %out, float addrspace(1)* noalias %in) nounwind {
  %val = load float, float addrspace(1)* %in, align 4
  %cvt = call i16 @llvm.convert.to.fp16.f32(float %val) nounwind readnone
  store i16 %cvt, i16 addrspace(1)* %out, align 2
  ret void
}
