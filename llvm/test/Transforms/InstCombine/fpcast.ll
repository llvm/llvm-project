; Test some floating point casting cases
; RUN: opt < %s -instcombine -S | FileCheck %s

define i8 @test1() {
        %x = fptoui float 2.550000e+02 to i8            ; <i8> [#uses=1]
        ret i8 %x
; CHECK: ret i8 -1
}

define i8 @test2() {
        %x = fptosi float -1.000000e+00 to i8           ; <i8> [#uses=1]
        ret i8 %x
; CHECK: ret i8 -1
}

; CHECK: test3
define half @test3(float %a) {
; CHECK: fptrunc
; CHECK: llvm.fabs.f16
  %b = call float @llvm.fabs.f32(float %a)
  %c = fptrunc float %b to half
  ret half %c
}

; CHECK: test4
define half @test4(float %a) {
; CHECK: fptrunc
; CHECK: fsub
  %b = fsub float -0.0, %a
  %c = fptrunc float %b to half
  ret half %c
}

; CHECK: test4-fast
define half @test4-fast(float %a) {
; CHECK: fptrunc
; CHECK: fsub fast
  %b = fsub fast float -0.0, %a
  %c = fptrunc float %b to half
  ret half %c
}

; CHECK: test5
define half @test5(float %a, float %b, float %c) {
; CHECK: fcmp ogt
; CHECK: fptrunc
; CHECK: select
; CHECK: half 0xH3C00
  %d = fcmp ogt float %a, %b
  %e = select i1 %d, float %c, float 1.0
  %f = fptrunc float %e to half
  ret half %f
}

declare float @llvm.fabs.f32(float) nounwind readonly

define <1 x float> @test6(<1 x double> %V) {
  %frem = frem <1 x double> %V, %V
  %trunc = fptrunc <1 x double> %frem to <1 x float>
  ret <1 x float> %trunc
; CHECK-LABEL: @test6
; CHECK-NEXT: %[[frem:.*]]  = frem <1 x double> %V, %V
; CHECK-NEXT: %[[trunc:.*]] = fptrunc <1 x double> %[[frem]] to <1 x float>
; CHECK-NEXT: ret <1 x float> %trunc
}

define float @test7(double %V) {
  %frem = frem double %V, 1.000000e+00
  %trunc = fptrunc double %frem to float
  ret float %trunc
; CHECK-LABEL: @test7
; CHECK-NEXT: %[[frem:.*]]  = frem double %V, 1.000000e+00
; CHECK-NEXT: %[[trunc:.*]] = fptrunc double %frem to float
; CHECK-NEXT: ret float %trunc
}

define float @test8(float %V) {
  %fext = fpext float %V to double
  %frem = frem double %fext, 1.000000e-01
  %trunc = fptrunc double %frem to float
  ret float %trunc
; CHECK-LABEL: @test8
; CHECK-NEXT: %[[fext:.*]]  = fpext float %V to double
; CHECK-NEXT: %[[frem:.*]]  = frem double %fext, 1.000000e-01
; CHECK-NEXT: %[[trunc:.*]] = fptrunc double %frem to float
; CHECK-NEXT: ret float %trunc
}

; CHECK-LABEL: @test_fptrunc_fptrunc
; CHECK-NOT: fptrunc double {{.*}} to half
define half @test_fptrunc_fptrunc(double %V) {
  %t1 = fptrunc double %V to float
  %t2 = fptrunc float %t1 to half
  ret half %t2
}
