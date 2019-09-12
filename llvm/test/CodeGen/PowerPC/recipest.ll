; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -mattr=-vsx | FileCheck %s

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

declare double @llvm.sqrt.f64(double)
declare float @llvm.sqrt.f32(float)
declare <4 x float> @llvm.sqrt.v4f32(<4 x float>)

define double @foo_fmf(double %a, double %b) nounwind {
; CHECK: @foo_fmf
; CHECK: frsqrte
; CHECK: fmul
; CHECK-NEXT: fmadd
; CHECK-NEXT: fmul
; CHECK-NEXT: fmul
; CHECK-NEXT: fmul
; CHECK-NEXT: fmadd
; CHECK-NEXT: fmul
; CHECK-NEXT: fmul
; CHECK-NEXT: fmul
; CHECK: blr
  %x = call fast double @llvm.sqrt.f64(double %b)
  %r = fdiv fast double %a, %x
  ret double %r
}

define double @foo_safe(double %a, double %b) nounwind {
; CHECK: @foo_safe
; CHECK: fsqrt
; CHECK: fdiv
; CHECK: blr
  %x = call double @llvm.sqrt.f64(double %b)
  %r = fdiv double %a, %x
  ret double %r
}

define double @no_estimate_refinement_f64(double %a, double %b) #0 {
; CHECK-LABEL: @no_estimate_refinement_f64
; CHECK: frsqrte
; CHECK-NOT: fmadd
; CHECK: fmul
; CHECK-NOT: fmadd
; CHECK: blr
  %x = call fast double @llvm.sqrt.f64(double %b)
  %r = fdiv fast double %a, %x
  ret double %r
}

define double @foof_fmf(double %a, float %b) nounwind {
; CHECK: @foof_fmf
; CHECK-DAG: frsqrtes
; CHECK: fmuls
; CHECK-NEXT: fmadds
; CHECK-NEXT: fmuls
; CHECK-NEXT: fmuls
; CHECK-NEXT: fmul
; CHECK-NEXT: blr
  %x = call fast float @llvm.sqrt.f32(float %b)
  %y = fpext float %x to double
  %r = fdiv fast double %a, %y
  ret double %r
}

define double @foof_safe(double %a, float %b) nounwind {
; CHECK: @foof_safe
; CHECK: fsqrts
; CHECK: fdiv
; CHECK: blr
  %x = call float @llvm.sqrt.f32(float %b)
  %y = fpext float %x to double
  %r = fdiv double %a, %y
  ret double %r
}

define float @food_fmf(float %a, double %b) nounwind {
; CHECK: @food_fmf
; CHECK-DAG: frsqrte
; CHECK: fmul
; CHECK-NEXT: fmadd
; CHECK-NEXT: fmul
; CHECK-NEXT: fmul
; CHECK-NEXT: fmul
; CHECK-NEXT: fmadd
; CHECK-NEXT: fmul
; CHECK-NEXT: fmul
; CHECK-NEXT: frsp
; CHECK-NEXT: fmuls
; CHECK-NEXT: blr
  %x = call fast double @llvm.sqrt.f64(double %b)
  %y = fptrunc double %x to float
  %r = fdiv fast float %a, %y
  ret float %r
}

define float @food_safe(float %a, double %b) nounwind {
; CHECK: @food_safe
; CHECK: fsqrt
; CHECK: fdivs
; CHECK: blr
  %x = call double @llvm.sqrt.f64(double %b)
  %y = fptrunc double %x to float
  %r = fdiv float %a, %y
  ret float %r
}

define float @goo_fmf(float %a, float %b) nounwind {
; CHECK: @goo_fmf
; CHECK-DAG: frsqrtes
; CHECK: fmuls
; CHECK-NEXT: fmadds
; CHECK-NEXT: fmuls
; CHECK-NEXT: fmuls
; CHECK-NEXT: fmuls
; CHECK-NEXT: blr
  %x = call fast float @llvm.sqrt.f32(float %b)
  %r = fdiv fast float %a, %x
  ret float %r
}

define float @goo_safe(float %a, float %b) nounwind {
; CHECK: @goo_safe
; CHECK: fsqrts
; CHECK: fdivs
; CHECK: blr
  %x = call float @llvm.sqrt.f32(float %b)
  %r = fdiv float %a, %x
  ret float %r
}

define float @no_estimate_refinement_f32(float %a, float %b) #0 {
; CHECK-LABEL: @no_estimate_refinement_f32
; CHECK: frsqrtes
; CHECK-NOT: fmadds
; CHECK: fmuls
; CHECK-NOT: fmadds
; CHECK: blr
  %x = call fast float @llvm.sqrt.f32(float %b)
  %r = fdiv fast float %a, %x
  ret float %r
}

; Recognize that this is rsqrt(a) * rcp(b) * c, 
; not 1 / ( 1 / sqrt(a)) * rcp(b) * c.
define float @rsqrt_fmul_fmf(float %a, float %b, float %c) {
; CHECK: @rsqrt_fmul_fmf
; CHECK-DAG: frsqrtes
; CHECK: fmuls
; CHECK-NEXT: fmadds
; CHECK-NEXT: fmuls
; CHECK-DAG: fres
; CHECK-COUNT-3: fmuls
; CHECK-NEXT: fmsubs
; CHECK-NEXT: fmadds
; CHECK-NEXT: fmuls
; CHECK-NEXT: blr
  %x = call fast float @llvm.sqrt.f32(float %a)
  %y = fmul fast float %x, %b 
  %z = fdiv fast float %c, %y
  ret float %z
}

; Recognize that this is rsqrt(a) * rcp(b) * c, 
; not 1 / ( 1 / sqrt(a)) * rcp(b) * c.
define float @rsqrt_fmul_safe(float %a, float %b, float %c) {
; CHECK: @rsqrt_fmul_safe
; CHECK: fsqrts
; CHECK: fmuls
; CHECK: fdivs
; CHECK: blr
  %x = call float @llvm.sqrt.f32(float %a)
  %y = fmul float %x, %b 
  %z = fdiv float %c, %y
  ret float %z
}

define <4 x float> @hoo_fmf(<4 x float> %a, <4 x float> %b) nounwind {
; CHECK: @hoo_fmf
; CHECK: vrsqrtefp
  %x = call fast <4 x float> @llvm.sqrt.v4f32(<4 x float> %b)
  %r = fdiv fast <4 x float> %a, %x
  ret <4 x float> %r
}

define <4 x float> @hoo_safe(<4 x float> %a, <4 x float> %b) nounwind {
; CHECK: @hoo_safe
; CHECK-NOT: vrsqrtefp
; CHECK: blr
  %x = call <4 x float> @llvm.sqrt.v4f32(<4 x float> %b)
  %r = fdiv <4 x float> %a, %x
  ret <4 x float> %r
}

define double @foo2_fmf(double %a, double %b) nounwind {
; CHECK: @foo2_fmf
; CHECK-DAG: fre
; CHECK-DAG: fnmsub
; CHECK: fmadd
; CHECK-NEXT: fmul
; CHECK-NEXT: fnmsub
; CHECK-NEXT: fmadd
; CHECK-NEXT: blr
  %r = fdiv fast double %a, %b
  ret double %r
}

define double @foo2_safe(double %a, double %b) nounwind {
; CHECK: @foo2_safe
; CHECK: fdiv
; CHECK: blr
  %r = fdiv double %a, %b
  ret double %r
}

define float @goo2_fmf(float %a, float %b) nounwind {
; CHECK: @goo2_fmf
; CHECK-DAG: fres
; CHECK-NEXT: fmuls
; CHECK-DAG: fnmsubs
; CHECK: fmadds
; CHECK-NEXT: blr
  %r = fdiv fast float %a, %b
  ret float %r
}

define float @goo2_safe(float %a, float %b) nounwind {
; CHECK: @goo2_safe
; CHECK: fdivs
; CHECK: blr
  %r = fdiv float %a, %b
  ret float %r
}

define <4 x float> @hoo2_fmf(<4 x float> %a, <4 x float> %b) nounwind {
; CHECK: @hoo2_fmf
; CHECK: vrefp
  %r = fdiv fast <4 x float> %a, %b
  ret <4 x float> %r
}

define <4 x float> @hoo2_safe(<4 x float> %a, <4 x float> %b) nounwind {
; CHECK: @hoo2_safe
; CHECK-NOT: vrefp
; CHECK: blr
  %r = fdiv <4 x float> %a, %b
  ret <4 x float> %r
}

define double @foo3_fmf(double %a) nounwind {
; CHECK: @foo3_fmf
; CHECK: fcmpu
; CHECK-DAG: frsqrte
; CHECK: fmul
; CHECK-NEXT: fmadd
; CHECK-NEXT: fmul
; CHECK-NEXT: fmul
; CHECK-NEXT: fmul
; CHECK-NEXT: fmadd
; CHECK-NEXT: fmul
; CHECK-NEXT: fmul
; CHECK: blr
  %r = call fast double @llvm.sqrt.f64(double %a)
  ret double %r
}

define double @foo3_safe(double %a) nounwind {
; CHECK: @foo3_safe
; CHECK: fsqrt
; CHECK: blr
  %r = call double @llvm.sqrt.f64(double %a)
  ret double %r
}

define float @goo3_fmf(float %a) nounwind {
; CHECK: @goo3_fmf
; CHECK: fcmpu
; CHECK-DAG: frsqrtes
; CHECK: fmuls
; CHECK-NEXT: fmadds
; CHECK-NEXT: fmuls
; CHECK-NEXT: fmuls
; CHECK: blr
  %r = call fast float @llvm.sqrt.f32(float %a)
  ret float %r
}

define float @goo3_safe(float %a) nounwind {
; CHECK: @goo3_safe
; CHECK: fsqrts
; CHECK: blr
  %r = call float @llvm.sqrt.f32(float %a)
  ret float %r
}

define <4 x float> @hoo3_fmf(<4 x float> %a) nounwind {
; CHECK: @hoo3_fmf
; CHECK: vrsqrtefp
; CHECK-DAG: vcmpeqfp
  %r = call fast <4 x float> @llvm.sqrt.v4f32(<4 x float> %a)
  ret <4 x float> %r
}

define <4 x float> @hoo3_safe(<4 x float> %a) nounwind {
; CHECK: @hoo3_safe
; CHECK-NOT: vrsqrtefp
; CHECK: blr
  %r = call <4 x float> @llvm.sqrt.v4f32(<4 x float> %a)
  ret <4 x float> %r
}

attributes #0 = { nounwind "reciprocal-estimates"="sqrtf:0,sqrtd:0" }
