; RUN: llc -mtriple=amdgpu6.00 < %s | FileCheck --check-prefix=GCN %s
; RUN: llc -mtriple=amdgpu7.04 < %s | FileCheck --check-prefix=GCN %s
; RUN: llc -mtriple=amdgpu8.02 -mattr=-flat-for-global < %s | FileCheck --check-prefix=GCN %s

declare float @llvm.fabs.f32(float) #0
declare float @llvm.floor.f32(float) #0

; GCN-LABEL: {{^}}fract_f32:
; GCN: v_floor_f32_e32 [[FLR:v[0-9]+]], [[INPUT:v[0-9]+]]
; GCN: v_sub_f32_e32 [[RESULT:v[0-9]+]], [[INPUT]], [[FLR]]

define float @fract_f32(float %src) #1 {
  %floor.x = call float @llvm.floor.f32(float %src)
  %fract = fsub float %src, %floor.x
  ret float %fract
}

; GCN-LABEL: {{^}}fract_f32_neg:
; GCN: v_floor_f32_e64 [[FLR:v[0-9]+]], -[[INPUT:v[0-9]+]]
; GCN: v_sub_f32_e64 [[RESULT:v[0-9]+]], -[[INPUT]], [[FLR]]
define float @fract_f32_neg(float %src) #1 {
  %x.neg = fsub float -0.0, %src
  %floor.x.neg = call float @llvm.floor.f32(float %x.neg)
  %fract = fsub float %x.neg, %floor.x.neg
  ret float %fract
}

; GCN-LABEL: {{^}}fract_f32_neg_abs:
; GCN: v_floor_f32_e64 [[FLR:v[0-9]+]], -|[[INPUT:v[0-9]+]]|
; GCN: v_sub_f32_e64 [[RESULT:v[0-9]+]], -|[[INPUT]]|, [[FLR]]
define float @fract_f32_neg_abs(float %src) #1 {
  %abs.x = call float @llvm.fabs.f32(float %src)
  %neg.abs.x = fsub float -0.0, %abs.x
  %floor.neg.abs.x = call float @llvm.floor.f32(float %neg.abs.x)
  %fract = fsub float %neg.abs.x, %floor.neg.abs.x
  ret float %fract
}

; GCN-LABEL: {{^}}multi_use_floor_fract_f32:
; GCN-DAG: v_floor_f32_e32 [[FLOOR:v[0-9]+]], [[INPUT:v[0-9]+]]
; GCN-DAG: v_sub_f32_e32 [[FRACT:v[0-9]+]], [[INPUT:v[0-9]+]]

; GCN: buffer_store_dword [[FLOOR]]
; GCN: buffer_store_dword [[FRACT]]
define amdgpu_kernel void @multi_use_floor_fract_f32(ptr addrspace(1) %out, ptr addrspace(1) %src) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %src.tid = getelementptr inbounds float, ptr addrspace(1) %src, i32 %tid
  %x = load float, ptr addrspace(1) %src.tid
  %floor.x = call float @llvm.floor.f32(float %x)
  %fract = fsub float %x, %floor.x
  store volatile float %floor.x, ptr addrspace(1) %out
  store volatile float %fract, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
