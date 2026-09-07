; RUN: llc -mtriple=aarch64-- -verify-machineinstrs -global-isel -stop-after=irtranslator -o - %s | FileCheck %s

declare float @sqrtf(float)
declare double @sqrt(double)

define float @sqrtf_libcall(float %x) {
  ; CHECK-LABEL: name: sqrtf_libcall
  ; CHECK: [[FSQRT:%[0-9]+]]:_(f32) = G_FSQRT
  ; CHECK-NOT: BL @sqrtf
  %result = call float @sqrtf(float %x) memory(none)
  ret float %result
}

define double @sqrt_libcall(double %x) {
  ; CHECK-LABEL: name: sqrt_libcall
  ; CHECK: [[FSQRT:%[0-9]+]]:_(f64) = nnan G_FSQRT
  ; CHECK-NOT: BL @sqrt
  %result = call nnan double @sqrt(double %x) memory(none)
  ret double %result
}

define double @sqrt_may_set_errno(double %x) {
  ; CHECK-LABEL: name: sqrt_may_set_errno
  ; CHECK: BL @sqrt
  %result = call double @sqrt(double %x)
  ret double %result
}
