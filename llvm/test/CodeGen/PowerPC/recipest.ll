; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -enable-unsafe-fp-math -mattr=-vsx | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -mattr=-vsx | FileCheck -check-prefix=CHECK-SAFE %s

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

declare double @llvm.sqrt.f64(double)
declare float @llvm.sqrt.f32(float)
declare <4 x float> @llvm.sqrt.v4f32(<4 x float>)

define double @foo(double %a, double %b) nounwind {
  %x = call double @llvm.sqrt.f64(double %b)
  %r = fdiv double %a, %x
  ret double %r

; CHECK: @foo
; CHECK-DAG: frsqrte
; CHECK-DAG: fnmsub
; CHECK: fmul
; CHECK-NEXT: fmadd
; CHECK-NEXT: fmul
; CHECK-NEXT: fmul
; CHECK-NEXT: fmadd
; CHECK-NEXT: fmul
; CHECK-NEXT: fmul
; CHECK: blr

; CHECK-SAFE: @foo
; CHECK-SAFE: fsqrt
; CHECK-SAFE: fdiv
; CHECK-SAFE: blr
}

define double @no_estimate_refinement_f64(double %a, double %b) #0 {
  %x = call double @llvm.sqrt.f64(double %b)
  %r = fdiv double %a, %x
  ret double %r

; CHECK-LABEL: @no_estimate_refinement_f64
; CHECK: frsqrte
; CHECK-NOT: fmadd
; CHECK: fmul
; CHECK-NOT: fmadd
; CHECK: blr
}


define double @foof(double %a, float %b) nounwind {
  %x = call float @llvm.sqrt.f32(float %b)
  %y = fpext float %x to double
  %r = fdiv double %a, %y
  ret double %r

; CHECK: @foof
; CHECK-DAG: frsqrtes
; CHECK-DAG: fnmsubs
; CHECK: fmuls
; CHECK-NEXT: fmadds
; CHECK-NEXT: fmuls
; CHECK-NEXT: fmul
; CHECK-NEXT: blr

; CHECK-SAFE: @foof
; CHECK-SAFE: fsqrts
; CHECK-SAFE: fdiv
; CHECK-SAFE: blr
}

define float @food(float %a, double %b) nounwind {
  %x = call double @llvm.sqrt.f64(double %b)
  %y = fptrunc double %x to float
  %r = fdiv float %a, %y
  ret float %r

; CHECK: @foo
; CHECK-DAG: frsqrte
; CHECK-DAG: fnmsub
; CHECK: fmul
; CHECK-NEXT: fmadd
; CHECK-NEXT: fmul
; CHECK-NEXT: fmul
; CHECK-NEXT: fmadd
; CHECK-NEXT: fmul
; CHECK-NEXT: frsp
; CHECK-NEXT: fmuls
; CHECK-NEXT: blr

; CHECK-SAFE: @foo
; CHECK-SAFE: fsqrt
; CHECK-SAFE: fdivs
; CHECK-SAFE: blr
}

define float @goo(float %a, float %b) nounwind {
  %x = call float @llvm.sqrt.f32(float %b)
  %r = fdiv float %a, %x
  ret float %r

; CHECK: @goo
; CHECK-DAG: frsqrtes
; CHECK-DAG: fnmsubs
; CHECK: fmuls
; CHECK-NEXT: fmadds
; CHECK-NEXT: fmuls
; CHECK-NEXT: fmuls
; CHECK-NEXT: blr

; CHECK-SAFE: @goo
; CHECK-SAFE: fsqrts
; CHECK-SAFE: fdivs
; CHECK-SAFE: blr
}


define float @no_estimate_refinement_f32(float %a, float %b) #0 {
  %x = call float @llvm.sqrt.f32(float %b)
  %r = fdiv float %a, %x
  ret float %r

; CHECK-LABEL: @no_estimate_refinement_f32
; CHECK: frsqrtes
; CHECK-NOT: fmadds
; CHECK: fmuls
; CHECK-NOT: fmadds
; CHECK: blr
}

; Recognize that this is rsqrt(a) * rcp(b) * c, 
; not 1 / ( 1 / sqrt(a)) * rcp(b) * c.
define float @rsqrt_fmul(float %a, float %b, float %c) {
  %x = call float @llvm.sqrt.f32(float %a)
  %y = fmul float %x, %b 
  %z = fdiv float %c, %y
  ret float %z

; CHECK: @rsqrt_fmul
; CHECK-DAG: frsqrtes
; CHECK-DAG: fres
; CHECK-DAG: fnmsubs
; CHECK-DAG: fmuls
; CHECK-DAG: fnmsubs
; CHECK-DAG: fmadds
; CHECK-DAG: fmadds
; CHECK: fmuls
; CHECK-NEXT: fmuls
; CHECK-NEXT: fmuls
; CHECK-NEXT: blr

; CHECK-SAFE: @rsqrt_fmul
; CHECK-SAFE: fsqrts
; CHECK-SAFE: fmuls
; CHECK-SAFE: fdivs
; CHECK-SAFE: blr
}

define <4 x float> @hoo(<4 x float> %a, <4 x float> %b) nounwind {
  %x = call <4 x float> @llvm.sqrt.v4f32(<4 x float> %b)
  %r = fdiv <4 x float> %a, %x
  ret <4 x float> %r

; CHECK: @hoo
; CHECK: vrsqrtefp

; CHECK-SAFE: @hoo
; CHECK-SAFE-NOT: vrsqrtefp
; CHECK-SAFE: blr
}

define double @foo2(double %a, double %b) nounwind {
  %r = fdiv double %a, %b
  ret double %r

; CHECK: @foo2
; CHECK-DAG: fre
; CHECK-DAG: fnmsub
; CHECK: fmadd
; CHECK-NEXT: fnmsub
; CHECK-NEXT: fmadd
; CHECK-NEXT: fmul
; CHECK-NEXT: blr

; CHECK-SAFE: @foo2
; CHECK-SAFE: fdiv
; CHECK-SAFE: blr
}

define float @goo2(float %a, float %b) nounwind {
  %r = fdiv float %a, %b
  ret float %r

; CHECK: @goo2
; CHECK-DAG: fres
; CHECK-DAG: fnmsubs
; CHECK: fmadds
; CHECK-NEXT: fmuls
; CHECK-NEXT: blr

; CHECK-SAFE: @goo2
; CHECK-SAFE: fdivs
; CHECK-SAFE: blr
}

define <4 x float> @hoo2(<4 x float> %a, <4 x float> %b) nounwind {
  %r = fdiv <4 x float> %a, %b
  ret <4 x float> %r

; CHECK: @hoo2
; CHECK: vrefp

; CHECK-SAFE: @hoo2
; CHECK-SAFE-NOT: vrefp
; CHECK-SAFE: blr
}

define double @foo3(double %a) nounwind {
  %r = call double @llvm.sqrt.f64(double %a)
  ret double %r

; CHECK: @foo3
; CHECK: fcmpu
; CHECK-DAG: frsqrte
; CHECK-DAG: fnmsub
; CHECK: fmul
; CHECK-NEXT: fmadd
; CHECK-NEXT: fmul
; CHECK-NEXT: fmul
; CHECK-NEXT: fmadd
; CHECK-NEXT: fmul
; CHECK-NEXT: fmul
; CHECK: blr

; CHECK-SAFE: @foo3
; CHECK-SAFE: fsqrt
; CHECK-SAFE: blr
}

define float @goo3(float %a) nounwind {
  %r = call float @llvm.sqrt.f32(float %a)
  ret float %r

; CHECK: @goo3
; CHECK: fcmpu
; CHECK-DAG: frsqrtes
; CHECK-DAG: fnmsubs
; CHECK: fmuls
; CHECK-NEXT: fmadds
; CHECK-NEXT: fmuls
; CHECK-NEXT: fmuls
; CHECK: blr

; CHECK-SAFE: @goo3
; CHECK-SAFE: fsqrts
; CHECK-SAFE: blr
}

define <4 x float> @hoo3(<4 x float> %a) nounwind {
  %r = call <4 x float> @llvm.sqrt.v4f32(<4 x float> %a)
  ret <4 x float> %r

; CHECK: @hoo3
; CHECK: vrsqrtefp
; CHECK-DAG: vcmpeqfp

; CHECK-SAFE: @hoo3
; CHECK-SAFE-NOT: vrsqrtefp
; CHECK-SAFE: blr
}

attributes #0 = { nounwind "reciprocal-estimates"="sqrtf:0,sqrtd:0" }

