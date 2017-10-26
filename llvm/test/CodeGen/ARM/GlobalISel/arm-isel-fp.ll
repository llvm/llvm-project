; RUN: llc -mtriple arm-linux-gnueabihf -mattr=+vfp2 -float-abi=hard -global-isel %s -o - | FileCheck %s -check-prefix CHECK -check-prefix HARD
; RUN: llc -mtriple arm-linux-gnueabi -mattr=+vfp2,+soft-float -float-abi=soft -global-isel %s -o - | FileCheck %s -check-prefix CHECK -check-prefix SOFT-AEABI
; RUN: llc -mtriple arm-linux-gnu- -mattr=+vfp2,+soft-float -float-abi=soft -global-isel %s -o - | FileCheck %s -check-prefix CHECK -check-prefix SOFT-DEFAULT

define arm_aapcscc float @test_frem_float(float %x, float %y) {
; CHECK-LABEL: test_frem_float:
; CHECK: bl fmodf
  %r = frem float %x, %y
  ret float %r
}

define arm_aapcscc double @test_frem_double(double %x, double %y) {
; CHECK-LABEL: test_frem_double:
; CHECK: bl fmod
  %r = frem double %x, %y
  ret double %r
}

declare float @llvm.pow.f32(float %x, float %y)
define arm_aapcscc float @test_fpow_float(float %x, float %y) {
; CHECK-LABEL: test_fpow_float:
; CHECK: bl powf
  %r = call float @llvm.pow.f32(float %x, float %y)
  ret float %r
}

declare double @llvm.pow.f64(double %x, double %y)
define arm_aapcscc double @test_fpow_double(double %x, double %y) {
; CHECK-LABEL: test_fpow_double:
; CHECK: bl pow
  %r = call double @llvm.pow.f64(double %x, double %y)
  ret double %r
}

define arm_aapcscc float @test_add_float(float %x, float %y) {
; CHECK-LABEL: test_add_float:
; HARD: vadd.f32
; SOFT-AEABI: bl __aeabi_fadd
; SOFT-DEFAULT: bl __addsf3
  %r = fadd float %x, %y
  ret float %r
}

define arm_aapcscc double @test_add_double(double %x, double %y) {
; CHECK-LABEL: test_add_double:
; HARD: vadd.f64
; SOFT-AEABI: bl __aeabi_dadd
; SOFT-DEFAULT: bl __adddf3
  %r = fadd double %x, %y
  ret double %r
}

define arm_aapcs_vfpcc i32 @test_cmp_float_ogt(float %x, float %y) {
; CHECK-LABEL: test_cmp_float_ogt
; HARD: vcmp.f32
; HARD: vmrs APSR_nzcv, fpscr
; HARD-NEXT: movgt
; SOFT-AEABI: bl __aeabi_fcmpgt
; SOFT-DEFAULT: bl __gtsf2
entry:
  %v = fcmp ogt float %x, %y
  %r = zext i1 %v to i32
  ret i32 %r
}

define arm_aapcs_vfpcc i32 @test_cmp_float_one(float %x, float %y) {
; CHECK-LABEL: test_cmp_float_one
; HARD: vcmp.f32
; HARD: vmrs APSR_nzcv, fpscr
; HARD: movgt
; HARD-NOT: vcmp
; HARD: movmi
; SOFT-AEABI-DAG: bl __aeabi_fcmpgt
; SOFT-AEABI-DAG: bl __aeabi_fcmplt
; SOFT-DEFAULT-DAG: bl __gtsf2
; SOFT-DEFAULT-DAG: bl __ltsf2
entry:
  %v = fcmp one float %x, %y
  %r = zext i1 %v to i32
  ret i32 %r
}
