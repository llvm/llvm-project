; RUN: llc < %s -mtriple=x86_64-apple-macosx -mattr=+sse41 | FileCheck -check-prefix=CHECK-SSE %s
; RUN: llc < %s -mtriple=x86_64-apple-macosx -mattr=+avx | FileCheck -check-prefix=CHECK-AVX %s

define float @test1(float %x) nounwind  {
  %call = tail call float @floorf(float %x) nounwind readnone
  ret float %call

; CHECK-SSE: test1:
; CHECK-SSE: roundss $1

; CHECK-AVX: test1:
; CHECK-AVX: vroundss $1
}

declare float @floorf(float) nounwind readnone

define double @test2(double %x) nounwind  {
  %call = tail call double @floor(double %x) nounwind readnone
  ret double %call

; CHECK-SSE: test2:
; CHECK-SSE: roundsd $1

; CHECK-AVX: test2:
; CHECK-AVX: vroundsd $1
}

declare double @floor(double) nounwind readnone

define float @test3(float %x) nounwind  {
  %call = tail call float @nearbyintf(float %x) nounwind readnone
  ret float %call

; CHECK-SSE: test3:
; CHECK-SSE: roundss $12

; CHECK-AVX: test3:
; CHECK-AVX: vroundss $12
}

declare float @nearbyintf(float) nounwind readnone

define double @test4(double %x) nounwind  {
  %call = tail call double @nearbyint(double %x) nounwind readnone
  ret double %call

; CHECK-SSE: test4:
; CHECK-SSE: roundsd $12

; CHECK-AVX: test4:
; CHECK-AVX: vroundsd $12
}

declare double @nearbyint(double) nounwind readnone

define float @test5(float %x) nounwind  {
  %call = tail call float @ceilf(float %x) nounwind readnone
  ret float %call

; CHECK-SSE: test5:
; CHECK-SSE: roundss $2

; CHECK-AVX: test5:
; CHECK-AVX: vroundss $2
}

declare float @ceilf(float) nounwind readnone

define double @test6(double %x) nounwind  {
  %call = tail call double @ceil(double %x) nounwind readnone
  ret double %call

; CHECK-SSE: test6:
; CHECK-SSE: roundsd $2

; CHECK-AVX: test6:
; CHECK-AVX: vroundsd $2
}

declare double @ceil(double) nounwind readnone

define float @test7(float %x) nounwind  {
  %call = tail call float @rintf(float %x) nounwind readnone
  ret float %call

; CHECK-SSE: test7:
; CHECK-SSE: roundss $4

; CHECK-AVX: test7:
; CHECK-AVX: vroundss $4
}

declare float @rintf(float) nounwind readnone

define double @test8(double %x) nounwind  {
  %call = tail call double @rint(double %x) nounwind readnone
  ret double %call

; CHECK-SSE: test8:
; CHECK-SSE: roundsd $4

; CHECK-AVX: test8:
; CHECK-AVX: vroundsd $4
}

declare double @rint(double) nounwind readnone

define float @test9(float %x) nounwind  {
  %call = tail call float @truncf(float %x) nounwind readnone
  ret float %call

; CHECK-SSE: test9:
; CHECK-SSE: roundss $3

; CHECK-AVX: test9:
; CHECK-AVX: vroundss $3
}

declare float @truncf(float) nounwind readnone

define double @test10(double %x) nounwind  {
  %call = tail call double @trunc(double %x) nounwind readnone
  ret double %call

; CHECK-SSE: test10:
; CHECK-SSE: roundsd $3

; CHECK-AVX: test10:
; CHECK-AVX: vroundsd $3
}

declare double @trunc(double) nounwind readnone
