; RUN: llc -O1 -march=arm -mattr=+vfp2 -mtriple=arm-linux-gnueabi < %s | FileCheck %s
; pr4939

define void @test(double* %x, double* %y) nounwind {
  %1 = load double, double* %x
  %2 = load double, double* %y
  %3 = fsub double -0.000000e+00, %1
  %4 = fcmp ugt double %2, %3
  br i1 %4, label %bb1, label %bb2

bb1:
;CHECK: vstrhi
  store double %1, double* %y
  br label %bb2

bb2:
  ret void
}
