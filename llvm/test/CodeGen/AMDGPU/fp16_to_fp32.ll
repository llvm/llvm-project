; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=EGCM -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cayman -verify-machineinstrs < %s | FileCheck -check-prefix=CM -check-prefix=EGCM -check-prefix=FUNC %s

declare float @llvm.convert.from.fp16.f32(i16) nounwind readnone

; FUNC-LABEL: {{^}}test_convert_fp16_to_fp32:
; GCN: buffer_load_ushort [[VAL:v[0-9]+]]
; GCN: v_cvt_f32_f16_e32 [[RESULT:v[0-9]+]], [[VAL]]
; GCN: buffer_store_dword [[RESULT]]

; EG: MEM_RAT_CACHELESS STORE_RAW [[RES:T[0-9]+\.[XYZW]]]
; CM: MEM_RAT_CACHELESS STORE_DWORD [[RES:T[0-9]+\.[XYZW]]]
; EGCM: VTX_READ_16 [[VAL:T[0-9]+\.[XYZW]]]
; EGCM: FLT16_TO_FLT32{{[ *]*}}[[RES]], [[VAL]]
define void @test_convert_fp16_to_fp32(float addrspace(1)* noalias %out, i16 addrspace(1)* noalias %in) nounwind {
  %val = load i16, i16 addrspace(1)* %in, align 2
  %cvt = call float @llvm.convert.from.fp16.f32(i16 %val) nounwind readnone
  store float %cvt, float addrspace(1)* %out, align 4
  ret void
}
