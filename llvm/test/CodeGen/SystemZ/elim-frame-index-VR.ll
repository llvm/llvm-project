; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s
;
; Test that a spill/reload of a VR32/VR64 reg uses the FP opcode supporting
; 20-bit displacement if needed and possible.

define void @f1(i32 %arg, ...)  {
; CHECK-LABEL: f1:
; CHECK-NOT: lay
; CHECK: stdy %f0, 4400(%r15)
bb:
  %i = alloca [4096 x i8]
  ret void
}

define void @f2(float %Arg) {
; CHECK-LABEL: f2:
; CHECK-NOT: lay
; CHECK: stey %f0, 4172(%r15)
bb:
  %i = alloca [1000 x float]
  %i2 = getelementptr inbounds [1000 x float], [1000 x float]* %i, i64 0, i64 999
  br i1 undef, label %bb3, label %bb2

bb2:
  store float %Arg , float* %i2
  br label %bb3

bb3:
  ret void
}

define void @f3(double* %Dst) {
; CHECK-LABEL: f3:
; CHECK-NOT: lay
; CHECK: ldy %f0, 4168(%r15)
bb:
  %i = alloca [500 x double]
  br i1 undef, label %bb3, label %bb2

bb2:
  %i12 = getelementptr inbounds [500 x double], [500 x double]* %i, i64 0, i64 499
  %i13 = load double, double* %i12
  %i14 = fdiv double %i13, 0.000000e+00
  store double %i14, double* %Dst
  br label %bb3

bb3:
  ret void
}

define void @f4(float* %Dst) {
; CHECK-LABEL: f4:
; CHECK-NOT: lay
; CHECK: ley %f0, 4172(%r15)
bb:
  %i = alloca [1000 x float]
  br i1 undef, label %bb3, label %bb2

bb2:
  %i12 = getelementptr inbounds [1000 x float], [1000 x float]* %i, i64 0, i64 999
  %i13 = load float, float* %i12
  %i14 = fdiv float %i13, 0.000000e+00
  store float %i14, float* %Dst
  br label %bb3

bb3:
  ret void
}
