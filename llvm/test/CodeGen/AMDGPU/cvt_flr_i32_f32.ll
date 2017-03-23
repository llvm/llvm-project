; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI-SAFE -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -enable-no-nans-fp-math -verify-machineinstrs < %s | FileCheck -check-prefix=SI-NONAN -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare float @llvm.fabs.f32(float) #1
declare float @llvm.floor.f32(float) #1

; FUNC-LABEL: {{^}}cvt_flr_i32_f32_0:
; SI-SAFE-NOT: v_cvt_flr_i32_f32
; SI-NOT: add
; SI-NONAN: v_cvt_flr_i32_f32_e32 v{{[0-9]+}}, s{{[0-9]+}}
; SI: s_endpgm
define amdgpu_kernel void @cvt_flr_i32_f32_0(i32 addrspace(1)* %out, float %x) #0 {
  %floor = call float @llvm.floor.f32(float %x) #1
  %cvt = fptosi float %floor to i32
  store i32 %cvt, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}cvt_flr_i32_f32_1:
; SI: v_add_f32_e64 [[TMP:v[0-9]+]], s{{[0-9]+}}, 1.0
; SI-SAFE-NOT: v_cvt_flr_i32_f32
; SI-NONAN: v_cvt_flr_i32_f32_e32 v{{[0-9]+}}, [[TMP]]
; SI: s_endpgm
define amdgpu_kernel void @cvt_flr_i32_f32_1(i32 addrspace(1)* %out, float %x) #0 {
  %fadd = fadd float %x, 1.0
  %floor = call float @llvm.floor.f32(float %fadd) #1
  %cvt = fptosi float %floor to i32
  store i32 %cvt, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}cvt_flr_i32_f32_fabs:
; SI-NOT: add
; SI-SAFE-NOT: v_cvt_flr_i32_f32
; SI-NONAN: v_cvt_flr_i32_f32_e64 v{{[0-9]+}}, |s{{[0-9]+}}|
; SI: s_endpgm
define amdgpu_kernel void @cvt_flr_i32_f32_fabs(i32 addrspace(1)* %out, float %x) #0 {
  %x.fabs = call float @llvm.fabs.f32(float %x) #1
  %floor = call float @llvm.floor.f32(float %x.fabs) #1
  %cvt = fptosi float %floor to i32
  store i32 %cvt, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}cvt_flr_i32_f32_fneg:
; SI-NOT: add
; SI-SAFE-NOT: v_cvt_flr_i32_f32
; SI-NONAN: v_cvt_flr_i32_f32_e64 v{{[0-9]+}}, -s{{[0-9]+}}
; SI: s_endpgm
define amdgpu_kernel void @cvt_flr_i32_f32_fneg(i32 addrspace(1)* %out, float %x) #0 {
  %x.fneg = fsub float -0.000000e+00, %x
  %floor = call float @llvm.floor.f32(float %x.fneg) #1
  %cvt = fptosi float %floor to i32
  store i32 %cvt, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}cvt_flr_i32_f32_fabs_fneg:
; SI-NOT: add
; SI-SAFE-NOT: v_cvt_flr_i32_f32
; SI-NONAN: v_cvt_flr_i32_f32_e64 v{{[0-9]+}}, -|s{{[0-9]+}}|
; SI: s_endpgm
define amdgpu_kernel void @cvt_flr_i32_f32_fabs_fneg(i32 addrspace(1)* %out, float %x) #0 {
  %x.fabs = call float @llvm.fabs.f32(float %x) #1
  %x.fabs.fneg = fsub float -0.000000e+00, %x.fabs
  %floor = call float @llvm.floor.f32(float %x.fabs.fneg) #1
  %cvt = fptosi float %floor to i32
  store i32 %cvt, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}no_cvt_flr_i32_f32_0:
; SI-NOT: v_cvt_flr_i32_f32
; SI: v_floor_f32
; SI: v_cvt_u32_f32_e32
; SI: s_endpgm
define amdgpu_kernel void @no_cvt_flr_i32_f32_0(i32 addrspace(1)* %out, float %x) #0 {
  %floor = call float @llvm.floor.f32(float %x) #1
  %cvt = fptoui float %floor to i32
  store i32 %cvt, i32 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
