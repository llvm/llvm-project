; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

define void @test_copysign_const_magnitude_d(double %X) {
; CHECK: [[SIGNMASK:L.+]]:
; CHECK-NEXT:   .quad -9223372036854775808    ## double -0
; CHECK-NEXT:   .quad 0                       ## double 0
; CHECK: [[ZERO:L.+]]:
; CHECK-NEXT:   .space 16
; CHECK: [[ONE:L.+]]:
; CHECK-NEXT:   .quad 4607182418800017408     ## double 1
; CHECK-NEXT:   .quad 0                       ## double 0
; CHECK-LABEL: test_copysign_const_magnitude_d:

; CHECK: id
  %iX = call double @id_d(double %X)

; CHECK-NEXT: andpd [[SIGNMASK]](%rip), %xmm0
  %d0 = call double @copysign(double 0.000000e+00, double %iX)

; CHECK-NEXT: id
  %id0 = call double @id_d(double %d0)

; CHECK-NEXT: andpd [[SIGNMASK]](%rip), %xmm0
; CHECK-NEXT: orpd [[ZERO]](%rip), %xmm0
  %dn0 = call double @copysign(double -0.000000e+00, double %id0)

; CHECK-NEXT: id
  %idn0 = call double @id_d(double %dn0)

; CHECK-NEXT: andpd [[SIGNMASK]](%rip), %xmm0
; CHECK-NEXT: orpd [[ONE]](%rip), %xmm0
  %d1 = call double @copysign(double 1.000000e+00, double %idn0)

; CHECK-NEXT: id
  %id1 = call double @id_d(double %d1)

; CHECK-NEXT: andpd [[SIGNMASK]](%rip), %xmm0
; CHECK-NEXT: orpd [[ONE]](%rip), %xmm0
  %dn1 = call double @copysign(double -1.000000e+00, double %id1)

; CHECK-NEXT: id
  %idn1 = call double @id_d(double %dn1)

; CHECK: retq
  ret void
}

define void @test_copysign_const_magnitude_f(float %X) {
; CHECK: [[SIGNMASK:L.+]]:
; CHECK-NEXT:   .long	2147483648              ## float -0
; CHECK-NEXT:   .long	0                       ## float 0
; CHECK-NEXT:   .long	0                       ## float 0
; CHECK-NEXT:   .long	0                       ## float 0
; CHECK: [[ZERO:L.+]]:
; CHECK-NEXT:   .space 16
; CHECK: [[ONE:L.+]]:
; CHECK-NEXT:   .long	1065353216              ## float 1
; CHECK-NEXT:   .long	0                       ## float 0
; CHECK-NEXT:   .long	0                       ## float 0
; CHECK-NEXT:   .long	0                       ## float 0
; CHECK-LABEL: test_copysign_const_magnitude_f:

; CHECK: id
  %iX = call float @id_f(float %X)

; CHECK-NEXT: andps [[SIGNMASK]](%rip), %xmm0
  %d0 = call float @copysignf(float 0.000000e+00, float %iX)

; CHECK-NEXT: id
  %id0 = call float @id_f(float %d0)

; CHECK-NEXT: andps [[SIGNMASK]](%rip), %xmm0
; CHECK-NEXT: orps [[ZERO]](%rip), %xmm0
  %dn0 = call float @copysignf(float -0.000000e+00, float %id0)

; CHECK-NEXT: id
  %idn0 = call float @id_f(float %dn0)

; CHECK-NEXT: andps [[SIGNMASK]](%rip), %xmm0
; CHECK-NEXT: orps [[ONE]](%rip), %xmm0
  %d1 = call float @copysignf(float 1.000000e+00, float %idn0)

; CHECK-NEXT: id
  %id1 = call float @id_f(float %d1)

; CHECK-NEXT: andps [[SIGNMASK]](%rip), %xmm0
; CHECK-NEXT: orps [[ONE]](%rip), %xmm0
  %dn1 = call float @copysignf(float -1.000000e+00, float %id1)

; CHECK-NEXT: id
  %idn1 = call float @id_f(float %dn1)

; CHECK: retq
  ret void
}

declare double @copysign(double, double) nounwind readnone
declare float @copysignf(float, float) nounwind readnone

; Dummy identity functions, so we always have xmm0, and prevent optimizations.
declare double @id_d(double)
declare float @id_f(float)
