; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN-SAFE -check-prefix=GCN -check-prefix=SI %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefix=GCN-SAFE -check-prefix=GCN -check-prefix=CI %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN-SAFE -check-prefix=GCN -check-prefix=FUNC %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -check-prefix=GCN-UNSAFE -check-prefix=GCN %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -check-prefix=GCN-UNSAFE -check-prefix=GCN %s

declare float @llvm.fabs.f32(float) #0
declare float @llvm.floor.f32(float) #0

; GCN-LABEL: {{^}}fract_f32:
; GCN-SAFE: v_floor_f32_e32 [[FLR:v[0-9]+]], [[INPUT:v[0-9]+]]
; GCN-SAFE: v_subrev_f32_e32 [[RESULT:v[0-9]+]], [[FLR]], [[INPUT]]

; GCN-UNSAFE: v_fract_f32_e32 [[RESULT:v[0-9]+]], [[INPUT:v[0-9]+]]

; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @fract_f32(float addrspace(1)* %out, float addrspace(1)* %src) #1 {
  %x = load float, float addrspace(1)* %src
  %floor.x = call float @llvm.floor.f32(float %x)
  %fract = fsub float %x, %floor.x
  store float %fract, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fract_f32_neg:
; GCN-SAFE: v_floor_f32_e64 [[FLR:v[0-9]+]], -[[INPUT:v[0-9]+]]
; GCN-SAFE: v_sub_f32_e64 [[RESULT:v[0-9]+]], -[[INPUT]], [[FLR]]

; GCN-UNSAFE: v_fract_f32_e64 [[RESULT:v[0-9]+]], -[[INPUT:v[0-9]+]]

; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @fract_f32_neg(float addrspace(1)* %out, float addrspace(1)* %src) #1 {
  %x = load float, float addrspace(1)* %src
  %x.neg = fsub float -0.0, %x
  %floor.x.neg = call float @llvm.floor.f32(float %x.neg)
  %fract = fsub float %x.neg, %floor.x.neg
  store float %fract, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fract_f32_neg_abs:
; GCN-SAFE: v_floor_f32_e64 [[FLR:v[0-9]+]], -|[[INPUT:v[0-9]+]]|
; GCN-SAFE: v_sub_f32_e64 [[RESULT:v[0-9]+]], -|[[INPUT]]|, [[FLR]]

; GCN-UNSAFE: v_fract_f32_e64 [[RESULT:v[0-9]+]], -|[[INPUT:v[0-9]+]]|

; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @fract_f32_neg_abs(float addrspace(1)* %out, float addrspace(1)* %src) #1 {
  %x = load float, float addrspace(1)* %src
  %abs.x = call float @llvm.fabs.f32(float %x)
  %neg.abs.x = fsub float -0.0, %abs.x
  %floor.neg.abs.x = call float @llvm.floor.f32(float %neg.abs.x)
  %fract = fsub float %neg.abs.x, %floor.neg.abs.x
  store float %fract, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}multi_use_floor_fract_f32:
; GCN-UNSAFE-DAG: v_floor_f32_e32 [[FLOOR:v[0-9]+]], [[INPUT:v[0-9]+]]
; GCN-UNSAFE-DAG: v_fract_f32_e32 [[FRACT:v[0-9]+]], [[INPUT:v[0-9]+]]

; GCN-UNSAFE: buffer_store_dword [[FLOOR]]
; GCN-UNSAFE: buffer_store_dword [[FRACT]]
define amdgpu_kernel void @multi_use_floor_fract_f32(float addrspace(1)* %out, float addrspace(1)* %src) #1 {
  %x = load float, float addrspace(1)* %src
  %floor.x = call float @llvm.floor.f32(float %x)
  %fract = fsub float %x, %floor.x
  store volatile float %floor.x, float addrspace(1)* %out
  store volatile float %fract, float addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
