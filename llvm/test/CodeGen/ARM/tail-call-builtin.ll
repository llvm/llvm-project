; RUN: llc -mtriple=thumbv7-linux-gnueabihf %s -o - | FileCheck %s

define arm_aapcs_vfpcc i64 @test_mismatched_call(double %in) {
; CHECK-LABEL: test_mismatched_call:
; CHECK: vmov r0, r1, d0
; CHECK: bl floor

  %val = tail call arm_aapcscc double @floor(double %in)
  %res = bitcast double %val to i64
  ret i64 %res
}

define arm_aapcs_vfpcc double @test_matched_call(double %in) {
; CHECK-LABEL: test_matched_call:
; CHECK: b _floor

  %val = tail call arm_aapcs_vfpcc double @_floor(double %in)
  ret double %val
}

define arm_aapcs_vfpcc void @test_irrelevant_call(double %in) {
; CHECK-LABEL: test_irrelevant_call:
; CHECK-NOT: bl floor

  %val = tail call arm_aapcscc double @floor(double %in)
  ret void
}

define arm_aapcs_vfpcc double @test_callingconv(double %in) {
; CHECK: test_callingconv:
; CHECK: bl floor

  %val = tail call arm_aapcscc double @floor(double %in)
  ret double %val
}

declare arm_aapcs_vfpcc double @_floor(double) nounwind readonly
declare arm_aapcscc double @floor(double) nounwind readonly
