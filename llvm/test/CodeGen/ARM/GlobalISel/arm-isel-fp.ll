; RUN: llc -mtriple arm-linux-gnueabihf -mattr=+vfp2 -float-abi=hard -global-isel %s -o - | FileCheck %s -check-prefix CHECK -check-prefix HARD
; RUN: llc -mtriple arm-linux-gnueabi -mattr=+vfp2,+soft-float -float-abi=soft -global-isel %s -o - | FileCheck %s -check-prefix CHECK -check-prefix SOFT-AEABI
; RUN: llc -mtriple arm-linux-gnu- -mattr=+vfp2,+soft-float -float-abi=soft -global-isel %s -o - | FileCheck %s -check-prefix CHECK -check-prefix SOFT-DEFAULT

define arm_aapcscc float @test_frem_float(float %x, float %y) {
; CHECK-LABEL: test_frem_float:
; CHECK: blx fmodf
  %r = frem float %x, %y
  ret float %r
}

define arm_aapcscc double @test_frem_double(double %x, double %y) {
; CHECK-LABEL: test_frem_double:
; CHECK: blx fmod
  %r = frem double %x, %y
  ret double %r
}

declare float @llvm.pow.f32(float %x, float %y)
define arm_aapcscc float @test_fpow_float(float %x, float %y) {
; CHECK-LABEL: test_fpow_float:
; CHECK: blx powf
  %r = call float @llvm.pow.f32(float %x, float %y)
  ret float %r
}

declare double @llvm.pow.f64(double %x, double %y)
define arm_aapcscc double @test_fpow_double(double %x, double %y) {
; CHECK-LABEL: test_fpow_double:
; CHECK: blx pow
  %r = call double @llvm.pow.f64(double %x, double %y)
  ret double %r
}

define arm_aapcscc float @test_add_float(float %x, float %y) {
; CHECK-LABEL: test_add_float:
; HARD: vadd.f32
; SOFT-AEABI: blx __aeabi_fadd
; SOFT-DEFAULT: blx __addsf3
  %r = fadd float %x, %y
  ret float %r
}

define arm_aapcscc double @test_add_double(double %x, double %y) {
; CHECK-LABEL: test_add_double:
; HARD: vadd.f64
; SOFT-AEABI: blx __aeabi_dadd
; SOFT-DEFAULT: blx __adddf3
  %r = fadd double %x, %y
  ret double %r
}
