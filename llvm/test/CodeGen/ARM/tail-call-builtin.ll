; RUN: llc -mtriple=thumbv7-linux-gnueabihf %s -o - | FileCheck %s

define i64 @test_mismatched_call(double %in) {
; CHECK-LABEL: test_mismatched_call:
; CHECK: bl floor
; CHECK: vmov r0, r1, d0

  %val = tail call double @floor(double %in)
  %res = bitcast double %val to i64
  ret i64 %res
}

define double @test_matched_call(double %in) {
; CHECK-LABEL: test_matched_call:
; CHECK: b floor

  %val = tail call double @floor(double %in)
  ret double %val
}

define void @test_irrelevant_call(double %in) {
; CHECK-LABEL: test_irrelevant_call:
; CHECK-NOT: bl floor

  %val = tail call double @floor(double %in)
  ret void
}

define arm_aapcscc double @test_callingconv(double %in) {
; CHECK: test_callingconv:
; CHECK: bl floor

  %val = tail call double @floor(double %in)
  ret double %val
}

declare double @floor(double) nounwind readonly
