; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=+avx < %s | FileCheck %s

define <4 x double> @test1(<4 x double> %A, <4 x double> %B) {
; CHECK-LABEL: test1:
; CHECK:       # BB#0:
; CHECK-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; CHECK-NEXT:    retq
entry:
  %0 = shufflevector <4 x double> %A, <4 x double> %B, <4 x i32> <i32 undef, i32 1, i32 undef, i32 5>
  ret <4 x double> %0
}

define <4 x double> @test2(<4 x double> %A, <4 x double> %B) {
; CHECK-LABEL: test2:
; CHECK:       # BB#0:
; CHECK-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; CHECK-NEXT:    retq
entry:
  %0 = shufflevector <4 x double> %A, <4 x double> %B, <4 x i32> <i32 undef, i32 1, i32 undef, i32 1>
  ret <4 x double> %0
}

define <4 x double> @test3(<4 x double> %A, <4 x double> %B) {
; CHECK-LABEL: test3:
; CHECK:       # BB#0:
; CHECK-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; CHECK-NEXT:    retq
entry:
  %0 = shufflevector <4 x double> %A, <4 x double> %B, <4 x i32> <i32 0, i32 1, i32 undef, i32 5>
  ret <4 x double> %0
}

define <4 x double> @test4(<4 x double> %A, <4 x double> %B) {
; CHECK-LABEL: test4:
; CHECK:       # BB#0:
; CHECK-NEXT:    vinsertf128 $1, %xmm0, %ymm0, %ymm0
; CHECK-NEXT:    retq
entry:
  %0 = shufflevector <4 x double> %A, <4 x double> %B, <4 x i32> <i32 0, i32 1, i32 undef, i32 1>
  ret <4 x double> %0
}
