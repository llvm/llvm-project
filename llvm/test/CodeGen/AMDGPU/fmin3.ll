; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9 %s

; GCN-LABEL: {{^}}test_fmin3_olt_0_f32:
; GCN: buffer_load_dword [[REGC:v[0-9]+]]
; GCN: buffer_load_dword [[REGB:v[0-9]+]]
; GCN: buffer_load_dword [[REGA:v[0-9]+]]
; GCN: v_min3_f32 [[RESULT:v[0-9]+]], [[REGC]], [[REGB]], [[REGA]]
; GCN: buffer_store_dword [[RESULT]],
define amdgpu_kernel void @test_fmin3_olt_0_f32(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #0 {
  %a = load volatile float, float addrspace(1)* %aptr, align 4
  %b = load volatile float, float addrspace(1)* %bptr, align 4
  %c = load volatile float, float addrspace(1)* %cptr, align 4
  %f0 = call float @llvm.minnum.f32(float %a, float %b)
  %f1 = call float @llvm.minnum.f32(float %f0, float %c)
  store float %f1, float addrspace(1)* %out, align 4
  ret void
}

; Commute operand of second fmin
; GCN-LABEL: {{^}}test_fmin3_olt_1_f32:
; GCN: buffer_load_dword [[REGB:v[0-9]+]]
; GCN: buffer_load_dword [[REGA:v[0-9]+]]
; GCN: buffer_load_dword [[REGC:v[0-9]+]]
; GCN: v_min3_f32 [[RESULT:v[0-9]+]], [[REGC]], [[REGB]], [[REGA]]
; GCN: buffer_store_dword [[RESULT]],
define amdgpu_kernel void @test_fmin3_olt_1_f32(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #0 {
  %a = load volatile float, float addrspace(1)* %aptr, align 4
  %b = load volatile float, float addrspace(1)* %bptr, align 4
  %c = load volatile float, float addrspace(1)* %cptr, align 4
  %f0 = call float @llvm.minnum.f32(float %a, float %b)
  %f1 = call float @llvm.minnum.f32(float %c, float %f0)
  store float %f1, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_fmin3_olt_0_f16:
; GCN: buffer_load_ushort [[REGC:v[0-9]+]]
; GCN: buffer_load_ushort [[REGB:v[0-9]+]]
; GCN: buffer_load_ushort [[REGA:v[0-9]+]]

; SI: v_min3_f32 [[RESULT_F32:v[0-9]+]],
; SI: v_cvt_f16_f32_e32 [[RESULT:v[0-9]+]], [[RESULT]]

; VI: v_min_f16_e32
; VI: v_min_f16_e32 [[RESULT:v[0-9]+]],

; GFX9: v_min3_f16 [[RESULT:v[0-9]+]], [[REGC]], [[REGB]], [[REGA]]
; GCN: buffer_store_short [[RESULT]],
define amdgpu_kernel void @test_fmin3_olt_0_f16(half addrspace(1)* %out, half addrspace(1)* %aptr, half addrspace(1)* %bptr, half addrspace(1)* %cptr) #0 {
  %a = load volatile half, half addrspace(1)* %aptr, align 2
  %b = load volatile half, half addrspace(1)* %bptr, align 2
  %c = load volatile half, half addrspace(1)* %cptr, align 2
  %f0 = call half @llvm.minnum.f16(half %a, half %b)
  %f1 = call half @llvm.minnum.f16(half %f0, half %c)
  store half %f1, half addrspace(1)* %out, align 2
  ret void
}

; Commute operand of second fmin
; GCN-LABEL: {{^}}test_fmin3_olt_1_f16:
; GCN: buffer_load_ushort [[REGA:v[0-9]+]]
; GCN: buffer_load_ushort [[REGB:v[0-9]+]]
; GCN: buffer_load_ushort [[REGC:v[0-9]+]]

; SI-DAG: v_cvt_f32_f16_e32 [[CVT_A:v[0-9]+]], [[REGA]]
; SI-DAG: v_cvt_f32_f16_e32 [[CVT_B:v[0-9]+]], [[REGB]]
; SI-DAG: v_cvt_f32_f16_e32 [[CVT_C:v[0-9]+]], [[REGC]]
; SI: v_min3_f32 [[RESULT_F32:v[0-9]+]], [[CVT_C]], [[CVT_A]], [[CVT_B]]
; SI: v_cvt_f16_f32_e32 [[RESULT:v[0-9]+]], [[RESULT_F32]]

; VI: v_min_f16_e32
; VI: v_min_f16_e32 [[RESULT:v[0-9]+]],

; GFX9: v_min3_f16 [[RESULT:v[0-9]+]], [[REGC]], [[REGA]], [[REGB]]
; GCN: buffer_store_short [[RESULT]],
define amdgpu_kernel void @test_fmin3_olt_1_f16(half addrspace(1)* %out, half addrspace(1)* %aptr, half addrspace(1)* %bptr, half addrspace(1)* %cptr) #0 {
  %a = load volatile half, half addrspace(1)* %aptr, align 2
  %b = load volatile half, half addrspace(1)* %bptr, align 2
  %c = load volatile half, half addrspace(1)* %cptr, align 2
  %f0 = call half @llvm.minnum.f16(half %a, half %b)
  %f1 = call half @llvm.minnum.f16(half %c, half %f0)
  store half %f1, half addrspace(1)* %out, align 2
  ret void
}

; Checks whether the test passes; performMinMaxCombine() should not optimize vector patterns of min3
; since there are no pack instructions for fmin3.
; GCN-LABEL: {{^}}no_fmin3_v2f16:

; SI: v_cvt_f16_f32_e32
; SI: v_min_f32_e32
; SI-NEXT: v_min_f32_e32
; SI-NEXT: v_min3_f32
; SI-NEXT: v_min3_f32

; VI: s_waitcnt
; VI-NEXT: v_min_f16_sdwa v4, v0, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-NEXT: v_min_f16_e32 v0, v0, v1
; VI-NEXT: v_min_f16_sdwa v1, v2, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; VI-NEXT: v_min_f16_e32 v0, v2, v0
; VI-NEXT: v_min_f16_sdwa v1, v1, v3 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
; VI-NEXT: v_min_f16_e32 v0, v0, v3
; VI-NEXT: v_or_b32_sdwa v0, v0, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
; VI-NEXT: s_setpc_b64

; GFX9: s_waitcnt
; GFX9-NEXT: v_pk_min_f16 v0, v0, v1
; GFX9-NEXT: v_pk_min_f16 v0, v2, v0
; GFX9-NEXT: v_pk_min_f16 v0, v0, v3
; GFX9-NEXT: s_setpc_b64
define <2 x half> @no_fmin3_v2f16(<2 x half> %a, <2 x half> %b, <2 x half> %c, <2 x half> %d) #2 {
entry:
  %min = call <2 x half> @llvm.minnum.v2f16(<2 x half> %a, <2 x half> %b)
  %min1 = call <2 x half> @llvm.minnum.v2f16(<2 x half> %c, <2 x half> %min)
  %res = call <2 x half> @llvm.minnum.v2f16(<2 x half> %min1, <2 x half> %d)
  ret <2 x half> %res
}

; GCN-LABEL: {{^}}test_fmin3_olt_0_f64:
; GCN-NOT: v_min3
define amdgpu_kernel void @test_fmin3_olt_0_f64(double addrspace(1)* %out, double addrspace(1)* %aptr, double addrspace(1)* %bptr, double addrspace(1)* %cptr) #0 {
  %a = load volatile double, double addrspace(1)* %aptr, align 4
  %b = load volatile double, double addrspace(1)* %bptr, align 4
  %c = load volatile double, double addrspace(1)* %cptr, align 4
  %f0 = call double @llvm.minnum.f64(double %a, double %b)
  %f1 = call double @llvm.minnum.f64(double %f0, double %c)
  store double %f1, double addrspace(1)* %out, align 4
  ret void
}

; Commute operand of second fmin
; GCN-LABEL: {{^}}test_fmin3_olt_1_f64:
; GCN-NOT: v_min3
define amdgpu_kernel void @test_fmin3_olt_1_f64(double addrspace(1)* %out, double addrspace(1)* %aptr, double addrspace(1)* %bptr, double addrspace(1)* %cptr) #0 {
  %a = load volatile double, double addrspace(1)* %aptr, align 4
  %b = load volatile double, double addrspace(1)* %bptr, align 4
  %c = load volatile double, double addrspace(1)* %cptr, align 4
  %f0 = call double @llvm.minnum.f64(double %a, double %b)
  %f1 = call double @llvm.minnum.f64(double %c, double %f0)
  store double %f1, double addrspace(1)* %out, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1
declare double @llvm.minnum.f64(double, double) #1
declare float @llvm.minnum.f32(float, float) #1
declare half @llvm.minnum.f16(half, half) #1
declare <2 x half> @llvm.minnum.v2f16(<2 x half>, <2 x half>)

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { nounwind "no-nans-fp-math"="true" }
