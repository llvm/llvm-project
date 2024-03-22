; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI-SAFE -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -mtriple=amdgcn -enable-no-nans-fp-math -verify-machineinstrs < %s | FileCheck -check-prefix=SI-NONAN -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga < %s | FileCheck -check-prefix=SI-SAFE -check-prefix=SI -check-prefix=FUNC %s

declare float @llvm.fabs.f32(float) nounwind readnone
declare float @llvm.floor.f32(float) nounwind readnone

; FUNC-LABEL: {{^}}cvt_rpi_i32_f32:
; SI-SAFE-NOT: v_cvt_rpi_i32_f32
; SI-NONAN: v_cvt_rpi_i32_f32_e32 v{{[0-9]+}}, s{{[0-9]+}}
; SI: s_endpgm
define amdgpu_kernel void @cvt_rpi_i32_f32(ptr addrspace(1) %out, float %x) nounwind {
  %fadd = fadd float %x, 0.5
  %floor = call float @llvm.floor.f32(float %fadd) nounwind readnone
  %cvt = fptosi float %floor to i32
  store i32 %cvt, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}cvt_rpi_i32_f32_fabs:
; SI-SAFE-NOT: v_cvt_rpi_i32_f32
; SI-NONAN: v_cvt_rpi_i32_f32_e64 v{{[0-9]+}}, |s{{[0-9]+}}|{{$}}
; SI: s_endpgm
define amdgpu_kernel void @cvt_rpi_i32_f32_fabs(ptr addrspace(1) %out, float %x) nounwind {
  %x.fabs = call float @llvm.fabs.f32(float %x) nounwind readnone
  %fadd = fadd float %x.fabs, 0.5
  %floor = call float @llvm.floor.f32(float %fadd) nounwind readnone
  %cvt = fptosi float %floor to i32
  store i32 %cvt, ptr addrspace(1) %out
  ret void
}

; FIXME: This doesn't work because it forms fsub 0.5, x
; FUNC-LABEL: {{^}}cvt_rpi_i32_f32_fneg:
; XSI-NONAN: v_cvt_rpi_i32_f32_e64 v{{[0-9]+}}, -s{{[0-9]+}}
; SI: v_sub_f32_e64 [[TMP:v[0-9]+]], 0.5, s{{[0-9]+}}
; SI-SAFE-NOT: v_cvt_flr_i32_f32
; SI-NONAN: v_cvt_flr_i32_f32_e32 {{v[0-9]+}}, [[TMP]]
; SI: s_endpgm
define amdgpu_kernel void @cvt_rpi_i32_f32_fneg(ptr addrspace(1) %out, float %x) nounwind {
  %x.fneg = fsub float -0.000000e+00, %x
  %fadd = fadd float %x.fneg, 0.5
  %floor = call float @llvm.floor.f32(float %fadd) nounwind readnone
  %cvt = fptosi float %floor to i32
  store i32 %cvt, ptr addrspace(1) %out
  ret void
}

; FIXME: This doesn't work for same reason as above
; FUNC-LABEL: {{^}}cvt_rpi_i32_f32_fabs_fneg:
; SI-SAFE-NOT: v_cvt_rpi_i32_f32
; XSI-NONAN: v_cvt_rpi_i32_f32_e64 v{{[0-9]+}}, -|s{{[0-9]+}}|

; SI: v_sub_f32_e64 [[TMP:v[0-9]+]], 0.5, |s{{[0-9]+}}|
; SI-SAFE-NOT: v_cvt_flr_i32_f32
; SI-NONAN: v_cvt_flr_i32_f32_e32 {{v[0-9]+}}, [[TMP]]
; SI: s_endpgm
define amdgpu_kernel void @cvt_rpi_i32_f32_fabs_fneg(ptr addrspace(1) %out, float %x) nounwind {
  %x.fabs = call float @llvm.fabs.f32(float %x) nounwind readnone
  %x.fabs.fneg = fsub float -0.000000e+00, %x.fabs
  %fadd = fadd float %x.fabs.fneg, 0.5
  %floor = call float @llvm.floor.f32(float %fadd) nounwind readnone
  %cvt = fptosi float %floor to i32
  store i32 %cvt, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}no_cvt_rpi_i32_f32_0:
; SI-NOT: v_cvt_rpi_i32_f32
; SI: v_add_f32
; SI: v_floor_f32
; SI: v_cvt_u32_f32
; SI: s_endpgm
define amdgpu_kernel void @no_cvt_rpi_i32_f32_0(ptr addrspace(1) %out, float %x) nounwind {
  %fadd = fadd float %x, 0.5
  %floor = call float @llvm.floor.f32(float %fadd) nounwind readnone
  %cvt = fptoui float %floor to i32
  store i32 %cvt, ptr addrspace(1) %out
  ret void
}
