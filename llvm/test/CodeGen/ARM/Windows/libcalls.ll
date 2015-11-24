; RUN: llc -mtriple thumbv7-windows-itanium -filetype asm -o - %s | FileCheck %s
; RUN: llc -mtriple thumbv7-windows-msvc -filetype asm -o - %s | FileCheck %s

define arm_aapcs_vfpcc i32 @sdiv32(i32 %divisor, i32 %divident) {
entry:
  %div = sdiv i32 %divident, %divisor
  ret i32 %div
}

; CHECK-LABEL: sdiv32
; CHECK: b __rt_sdiv

define arm_aapcs_vfpcc i64 @sdiv64(i64 %divisor, i64 %divident) {
entry:
  %div = sdiv i64 %divident, %divisor
  ret i64 %div
}

; CHECK-LABEL: sdiv64
; CHECK: bl __rt_sdiv64

define arm_aapcs_vfpcc i64 @stoi64(float %f) {
entry:
  %conv = fptosi float %f to i64
  ret i64 %conv
}

; CHECK-LABEL: stoi64
; CHECK: bl __stoi64

define arm_aapcs_vfpcc i64 @stou64(float %f) {
entry:
  %conv = fptoui float %f to i64
  ret i64 %conv
}

; CHECK-LABEL: stou64
; CHECK: bl __stou64

define arm_aapcs_vfpcc float @i64tos(i64 %i64) {
entry:
  %conv = sitofp i64 %i64 to float
  ret float %conv
}

; CHECK-LABEL: i64tos
; CHECK: bl __i64tos

define arm_aapcs_vfpcc float @u64tos(i64 %u64) {
entry:
  %conv = uitofp i64 %u64 to float
  ret float %conv
}

; CHECK-LABEL: u64tos
; CHECK: bl __u64tos

define arm_aapcs_vfpcc i64 @dtoi64(double %d) {
entry:
  %conv = fptosi double %d to i64
  ret i64 %conv
}

; CHECK-LABEL: dtoi64
; CHECK: bl __dtoi64

define arm_aapcs_vfpcc i64 @dtou64(double %d) {
entry:
  %conv = fptoui double %d to i64
  ret i64 %conv
}

; CHECK-LABEL: dtou64
; CHECK: bl __dtou64

define arm_aapcs_vfpcc double @i64tod(i64 %i64) {
entry:
  %conv = sitofp i64 %i64 to double
  ret double %conv
}

; CHECK-LABEL: i64tod
; CHECK: bl __i64tod

define arm_aapcs_vfpcc double @u64tod(i64 %i64) {
entry:
  %conv = uitofp i64 %i64 to double
  ret double %conv
}

; CHECK-LABEL: u64tod
; CHECK: bl __u64tod

