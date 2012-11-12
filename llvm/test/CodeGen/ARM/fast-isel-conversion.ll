; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=dynamic-no-pic -mtriple=armv7-apple-ios | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-ios | FileCheck %s --check-prefix=THUMB

; Test sitofp

define void @sitofp_single_i32(i32 %a, float %b) nounwind ssp {
entry:
; ARM: sitofp_single_i32
; ARM: vmov s0, r0
; ARM: vcvt.f32.s32 s0, s0
; THUMB: sitofp_single_i32
; THUMB: vmov s0, r0
; THUMB: vcvt.f32.s32 s0, s0
  %b.addr = alloca float, align 4
  %conv = sitofp i32 %a to float
  store float %conv, float* %b.addr, align 4
  ret void
}

define void @sitofp_single_i16(i16 %a, float %b) nounwind ssp {
entry:
; ARM: sitofp_single_i16
; ARM: sxth r0, r0
; ARM: vmov s0, r0
; ARM: vcvt.f32.s32 s0, s0
; THUMB: sitofp_single_i16
; THUMB: sxth r0, r0
; THUMB: vmov s0, r0
; THUMB: vcvt.f32.s32 s0, s0
  %b.addr = alloca float, align 4
  %conv = sitofp i16 %a to float
  store float %conv, float* %b.addr, align 4
  ret void
}

define void @sitofp_single_i8(i8 %a) nounwind ssp {
entry:
; ARM: sitofp_single_i8
; ARM: sxtb r0, r0
; ARM: vmov s0, r0
; ARM: vcvt.f32.s32 s0, s0
; THUMB: sitofp_single_i8
; THUMB: sxtb r0, r0
; THUMB: vmov s0, r0
; THUMB: vcvt.f32.s32 s0, s0
  %b.addr = alloca float, align 4
  %conv = sitofp i8 %a to float
  store float %conv, float* %b.addr, align 4
  ret void
}

define void @sitofp_double_i32(i32 %a, double %b) nounwind ssp {
entry:
; ARM: sitofp_double_i32
; ARM: vmov s0, r0
; ARM: vcvt.f64.s32 d16, s0
; THUMB: sitofp_double_i32
; THUMB: vmov s0, r0
; THUMB: vcvt.f64.s32 d16, s0
  %b.addr = alloca double, align 8
  %conv = sitofp i32 %a to double
  store double %conv, double* %b.addr, align 8
  ret void
}

define void @sitofp_double_i16(i16 %a, double %b) nounwind ssp {
entry:
; ARM: sitofp_double_i16
; ARM: sxth r0, r0
; ARM: vmov s0, r0
; ARM: vcvt.f64.s32 d16, s0
; THUMB: sitofp_double_i16
; THUMB: sxth r0, r0
; THUMB: vmov s0, r0
; THUMB: vcvt.f64.s32 d16, s0
  %b.addr = alloca double, align 8
  %conv = sitofp i16 %a to double
  store double %conv, double* %b.addr, align 8
  ret void
}

define void @sitofp_double_i8(i8 %a, double %b) nounwind ssp {
entry:
; ARM: sitofp_double_i8
; ARM: sxtb r0, r0
; ARM: vmov s0, r0
; ARM: vcvt.f64.s32 d16, s0
; THUMB: sitofp_double_i8
; THUMB: sxtb r0, r0
; THUMB: vmov s0, r0
; THUMB: vcvt.f64.s32 d16, s0
  %b.addr = alloca double, align 8
  %conv = sitofp i8 %a to double
  store double %conv, double* %b.addr, align 8
  ret void
}

; Test uitofp

define void @uitofp_single_i32(i32 %a, float %b) nounwind ssp {
entry:
; ARM: uitofp_single_i32
; ARM: vmov s0, r0
; ARM: vcvt.f32.u32 s0, s0
; THUMB: uitofp_single_i32
; THUMB: vmov s0, r0
; THUMB: vcvt.f32.u32 s0, s0
  %b.addr = alloca float, align 4
  %conv = uitofp i32 %a to float
  store float %conv, float* %b.addr, align 4
  ret void
}

define void @uitofp_single_i16(i16 %a, float %b) nounwind ssp {
entry:
; ARM: uitofp_single_i16
; ARM: uxth r0, r0
; ARM: vmov s0, r0
; ARM: vcvt.f32.u32 s0, s0
; THUMB: uitofp_single_i16
; THUMB: uxth r0, r0
; THUMB: vmov s0, r0
; THUMB: vcvt.f32.u32 s0, s0
  %b.addr = alloca float, align 4
  %conv = uitofp i16 %a to float
  store float %conv, float* %b.addr, align 4
  ret void
}

define void @uitofp_single_i8(i8 %a) nounwind ssp {
entry:
; ARM: uitofp_single_i8
; ARM: uxtb r0, r0
; ARM: vmov s0, r0
; ARM: vcvt.f32.u32 s0, s0
; THUMB: uitofp_single_i8
; THUMB: uxtb r0, r0
; THUMB: vmov s0, r0
; THUMB: vcvt.f32.u32 s0, s0
  %b.addr = alloca float, align 4
  %conv = uitofp i8 %a to float
  store float %conv, float* %b.addr, align 4
  ret void
}

define void @uitofp_double_i32(i32 %a, double %b) nounwind ssp {
entry:
; ARM: uitofp_double_i32
; ARM: vmov s0, r0
; ARM: vcvt.f64.u32 d16, s0
; THUMB: uitofp_double_i32
; THUMB: vmov s0, r0
; THUMB: vcvt.f64.u32 d16, s0
  %b.addr = alloca double, align 8
  %conv = uitofp i32 %a to double
  store double %conv, double* %b.addr, align 8
  ret void
}

define void @uitofp_double_i16(i16 %a, double %b) nounwind ssp {
entry:
; ARM: uitofp_double_i16
; ARM: uxth r0, r0
; ARM: vmov s0, r0
; ARM: vcvt.f64.u32 d16, s0
; THUMB: uitofp_double_i16
; THUMB: uxth r0, r0
; THUMB: vmov s0, r0
; THUMB: vcvt.f64.u32 d16, s0
  %b.addr = alloca double, align 8
  %conv = uitofp i16 %a to double
  store double %conv, double* %b.addr, align 8
  ret void
}

define void @uitofp_double_i8(i8 %a, double %b) nounwind ssp {
entry:
; ARM: uitofp_double_i8
; ARM: uxtb r0, r0
; ARM: vmov s0, r0
; ARM: vcvt.f64.u32 d16, s0
; THUMB: uitofp_double_i8
; THUMB: uxtb r0, r0
; THUMB: vmov s0, r0
; THUMB: vcvt.f64.u32 d16, s0
  %b.addr = alloca double, align 8
  %conv = uitofp i8 %a to double
  store double %conv, double* %b.addr, align 8
  ret void
}

; Test fptosi

define void @fptosi_float(float %a) nounwind ssp {
entry:
; ARM: fptosi_float
; ARM: vcvt.s32.f32 s0, s0
; THUMB: fptosi_float
; THUMB: vcvt.s32.f32 s0, s0
  %b.addr = alloca i32, align 4
  %conv = fptosi float %a to i32
  store i32 %conv, i32* %b.addr, align 4
  ret void
}

define void @fptosi_double(double %a) nounwind ssp {
entry:
; ARM: fptosi_double
; ARM: vcvt.s32.f64 s0, d16
; THUMB: fptosi_double
; THUMB: vcvt.s32.f64 s0, d16
  %b.addr = alloca i32, align 8
  %conv = fptosi double %a to i32
  store i32 %conv, i32* %b.addr, align 8
  ret void
}

; Test fptoui

define void @fptoui_float(float %a) nounwind ssp {
entry:
; ARM: fptoui_float
; ARM: vcvt.u32.f32 s0, s0
; THUMB: fptoui_float
; THUMB: vcvt.u32.f32 s0, s0
  %b.addr = alloca i32, align 4
  %conv = fptoui float %a to i32
  store i32 %conv, i32* %b.addr, align 4
  ret void
}

define void @fptoui_double(double %a) nounwind ssp {
entry:
; ARM: fptoui_double
; ARM: vcvt.u32.f64 s0, d16
; THUMB: fptoui_double
; THUMB: vcvt.u32.f64 s0, d16
  %b.addr = alloca i32, align 8
  %conv = fptoui double %a to i32
  store i32 %conv, i32* %b.addr, align 8
  ret void
}
