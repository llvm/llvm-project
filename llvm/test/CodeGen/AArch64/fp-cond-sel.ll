; RUN: llc -verify-machineinstrs -o - %s -mtriple=aarch64-none-linux-gnu -mcpu=cyclone | FileCheck %s

@varfloat = global float 0.0
@vardouble = global double 0.0

declare void @use_float(float)
declare void @use_double(double)

define void @test_csel(i32 %lhs32, i32 %rhs32, i64 %lhs64) {
; CHECK-LABEL: test_csel:

  %tst1 = icmp ugt i32 %lhs32, %rhs32
  %val1 = select i1 %tst1, float 0.0, float 1.0
  store float %val1, float* @varfloat
; CHECK: movi v[[FLT0:[0-9]+]].2d, #0
; CHECK: fmov s[[FLT1:[0-9]+]], #1.0
; CHECK: fcsel {{s[0-9]+}}, s[[FLT0]], s[[FLT1]], hi

  %rhs64 = sext i32 %rhs32 to i64
  %tst2 = icmp sle i64 %lhs64, %rhs64
  %val2 = select i1 %tst2, double 1.0, double 0.0
  store double %val2, double* @vardouble
; FLT0 is reused from above on ARM64.
; CHECK: fmov d[[FLT1:[0-9]+]], #1.0
; CHECK: fcsel {{d[0-9]+}}, d[[FLT1]], d[[FLT0]], le

  call void @use_float(float 0.0)
  call void @use_float(float 1.0)

  call void @use_double(double 0.0)
  call void @use_double(double 1.0)

  ret void
; CHECK: ret
}
