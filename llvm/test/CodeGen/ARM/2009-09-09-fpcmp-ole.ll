; RUN: llc -O1 -mattr=+vfp2 -mtriple=arm-linux-gnueabi < %s | FileCheck %s
; pr4939

define void @test(ptr %x, ptr %y) nounwind {
  %1 = load double, ptr %x
  %2 = load double, ptr %y
  %3 = fsub double -0.000000e+00, %1
  %4 = fcmp ugt double %2, %3
  br i1 %4, label %bb1, label %bb2

bb1:
;CHECK: vstrhi
  store double %1, ptr %y
  br label %bb2

bb2:
  ret void
}
