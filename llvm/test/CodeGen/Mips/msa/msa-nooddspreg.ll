; RUN: llc -march=mips -mcpu=mips32r5 -mattr=+fp64,+msa,+nooddspreg < %s | FileCheck %s

; Test that the register allocator honours +nooddspreg and does not pick an odd
; single precision subregister of an MSA register.

@f1 = external global float

@f2 = external global float

@v3 = external global <4 x float>

@d1 = external global double

define void @test() {
; CHECK-LABEL: test:
entry:
; CHECK-NOT: lwc1 $f{{[13579]+}}
; CHECK: lwc1 $f{{[02468]+}}
  %0 = load float, float * @f1
  %1 = insertelement <4 x float> undef,    float %0, i32 0
  %2 = insertelement <4 x float> %1,    float %0, i32 1
  %3 = insertelement <4 x float> %2,    float %0, i32 2
  %4 = insertelement <4 x float> %3,    float %0, i32 3

; CHECK-NOT: lwc1 $f{{[13579]+}}
; CHECK: lwc1 $f{{[02468]+}}
  %5 = load float, float * @f2
  %6 = insertelement <4 x float> undef,    float %5, i32 0
  %7 = insertelement <4 x float> %6,    float %5, i32 1
  %8 = insertelement <4 x float> %7,    float %5, i32 2
  %9 = insertelement <4 x float> %8,    float %5, i32 3

  %10 = fadd <4 x float> %4, %9
  store <4 x float> %10, <4 x float> * @v3
  ret void
}

; Test that the register allocator hnours +noodspreg and does not pick an odd
; single precision register for a load to perform a conversion to a double.

define void @test2() {
; CHECK-LABEL: test2:
entry:
; CHECK-NOT: lwc1 $f{{[13579]+}}
; CHECK: lwc1 $f{{[02468]+}}
  %0 = load float, float * @f1
  %1 = fpext float %0 to double
; CHECK-NOT: lwc1 $f{{[13579]+}}
; CHECK: lwc1 $f{{[02468]+}}
  %2 = load float, float * @f2
  %3 = fpext float %2 to double
  %4 = fadd double %1, %3
  store double%4, double * @d1
  ret void
}
