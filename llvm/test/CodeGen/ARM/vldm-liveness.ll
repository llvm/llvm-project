; RUN: llc -mtriple thumbv7-apple-ios -verify-machineinstrs -o - %s | FileCheck %s

; Make sure we emit the loads in ascending order, and form a vldmia.
;
; See vldm-liveness.mir for the bug this file originally testing.

define arm_aapcs_vfpcc <4 x float> @foo(float* %ptr) {
; CHECK-LABEL: foo:
; CHECK: vldmia r0, {s0, s1}
; CHECK: vldr s3, [r0, #8]
; CHECK: vldr s2, [r0, #16]
   %off0 = getelementptr float, float* %ptr, i32 0
   %val0 = load float, float* %off0
   %off1 = getelementptr float, float* %ptr, i32 1
   %val1 = load float, float* %off1
   %off4 = getelementptr float, float* %ptr, i32 4
   %val4 = load float, float* %off4
   %off2 = getelementptr float, float* %ptr, i32 2
   %val2 = load float, float* %off2

   %vec1 = insertelement <4 x float> undef, float %val0, i32 0
   %vec2 = insertelement <4 x float> %vec1, float %val1, i32 1
   %vec3 = insertelement <4 x float> %vec2, float %val4, i32 2
   %vec4 = insertelement <4 x float> %vec3, float %val2, i32 3

   ret <4 x float> %vec4
}
