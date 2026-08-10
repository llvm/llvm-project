; RUN: llc -mtriple=thumbv7em-none-macho %s -o - -mcpu=cortex-m4 | FileCheck --check-prefix=CHECK-HARD %s
; RUN: llc -mtriple=thumbv7m-none-macho %s -o - -mcpu=cortex-m4 | FileCheck --check-prefix=CHECK-SOFT %s
; RUN: llc -mtriple=thumbv7em-linux-gnueabi %s -o - -mcpu=cortex-m4 | FileCheck --check-prefix=CHECK-SOFT %s

define float @test_default_cc(float %a, float %b) {
; CHECK-HARD-LABEL: test_default_cc:
; CHECK-HARD-NOT: vmov
; CHECK-HARD: vadd.f32 s0, s0, s1
; CHECK-HARD-NOT: vmov

; CHECK-SOFT-LABEL: test_default_cc:
; CHECK-SOFT-DAG: vmov [[A:s[0-9]+]], r0
; CHECK-SOFT-DAG: vmov [[B:s[0-9]+]], r1
; CHECK-SOFT: vadd.f32 [[RES:s[0-9]+]], [[A]], [[B]]
; CHECK-SOFT: vmov r0, [[RES]]

  %res = fadd float %a, %b
  ret float %res
}


define arm_aapcs_vfpcc float @test_libcall(float %in) {
; CHECK-HARD-LABEL: test_libcall:
; CHECK-HARD-NOT: vmov
; CHECK-HARD: b.w _sinf

; CHECK-SOFT-LABEL: test_libcall:
; CHECK-SOFT: vmov r0, s0
; CHECK-SOFT: bl {{_?}}sinf
; CHECK-SOFT: vmov s0, r0

  %res = call float @llvm.sin.f32(float %in)
  ret float %res
}


declare float @llvm.sin.f32(float)
