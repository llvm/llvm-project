; RUN: opt -passes=replace-with-veclib -vector-library=LIBMVEC -S %s | FileCheck %s
; The same module contains callers with different target features. Matching a
; vector width alone is insufficient: class d requires AVX2, not just AVX.
target triple = "x86_64-unknown-linux-gnu"

define <4 x double> @generic(<4 x double> %x, <4 x double> %y) #0 {
; CHECK-LABEL: define <4 x double> @generic(
; CHECK-NEXT: [[R:%.*]] = call fast <4 x double> @llvm.pow.v4f64(<4 x double> %x, <4 x double> %y)
; CHECK-NEXT: ret <4 x double> [[R]]
  %r = call fast <4 x double> @llvm.pow.v4f64(<4 x double> %x, <4 x double> %y)
  ret <4 x double> %r
}

define <4 x double> @avx(<4 x double> %x, <4 x double> %y) #1 {
; CHECK-LABEL: define <4 x double> @avx(
; CHECK-NEXT: [[R:%.*]] = call fast <4 x double> @llvm.pow.v4f64(<4 x double> %x, <4 x double> %y)
; CHECK-NEXT: ret <4 x double> [[R]]
  %r = call fast <4 x double> @llvm.pow.v4f64(<4 x double> %x, <4 x double> %y)
  ret <4 x double> %r
}

define <4 x double> @avx2(<4 x double> %x, <4 x double> %y) #2 {
; CHECK-LABEL: define <4 x double> @avx2(
; CHECK-NEXT: [[R:%.*]] = call fast <4 x double> @_ZGVdN4vv_pow(<4 x double> %x, <4 x double> %y)
; CHECK-NEXT: ret <4 x double> [[R]]
  %r = call fast <4 x double> @llvm.pow.v4f64(<4 x double> %x, <4 x double> %y)
  ret <4 x double> %r
}

define <4 x double> @disabled_avx2(<4 x double> %x, <4 x double> %y) #5 {
; CHECK-LABEL: define <4 x double> @disabled_avx2(
; CHECK-NEXT: [[R:%.*]] = call fast <4 x double> @llvm.pow.v4f64(<4 x double> %x, <4 x double> %y)
; CHECK-NEXT: ret <4 x double> [[R]]
  %r = call fast <4 x double> @llvm.pow.v4f64(<4 x double> %x, <4 x double> %y)
  ret <4 x double> %r
}

define <4 x double> @prefer128(<4 x double> %x, <4 x double> %y) #6 {
; CHECK-LABEL: define <4 x double> @prefer128(
; CHECK-NEXT: [[R:%.*]] = call fast <4 x double> @_ZGVdN4vv_pow(<4 x double> %x, <4 x double> %y)
; CHECK-NEXT: ret <4 x double> [[R]]
  %r = call fast <4 x double> @llvm.pow.v4f64(<4 x double> %x, <4 x double> %y)
  ret <4 x double> %r
}

define <2 x double> @sse2(<2 x double> %x, <2 x double> %y) #0 {
; CHECK-LABEL: define <2 x double> @sse2(
; CHECK-NEXT: [[R:%.*]] = call fast <2 x double> @_ZGVbN2vv_pow(<2 x double> %x, <2 x double> %y)
; CHECK-NEXT: ret <2 x double> [[R]]
  %r = call fast <2 x double> @llvm.pow.v2f64(<2 x double> %x, <2 x double> %y)
  ret <2 x double> %r
}
declare <2 x double> @llvm.pow.v2f64(<2 x double>, <2 x double>)
declare <4 x double> @llvm.pow.v4f64(<4 x double>, <4 x double>)
attributes #0 = { "target-cpu"="x86-64" }
attributes #1 = { "target-cpu"="sandybridge" }
attributes #2 = { "target-cpu"="haswell" }
attributes #5 = { "target-cpu"="haswell" "target-features"="-avx2" }
attributes #6 = { "target-cpu"="haswell" "prefer-vector-width"="128" }
