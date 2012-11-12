; RUN: opt < %s -instcombine -S | FileCheck %s

; PR4374
define float @test1(float %a, float %b) nounwind {
  %t1 = fsub float %a, %b
  %t2 = fsub float -0.000000e+00, %t1

; CHECK:       %t1 = fsub float %a, %b
; CHECK-NEXT:  %t2 = fsub float -0.000000e+00, %t1

  ret float %t2
}

; <rdar://problem/7530098>
define double @test2(double %x, double %y) nounwind {
  %t1 = fadd double %x, %y
  %t2 = fsub double %x, %t1

; CHECK:      %t1 = fadd double %x, %y
; CHECK-NEXT: %t2 = fsub double %x, %t1

  ret double %t2
}
