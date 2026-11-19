; RUN: opt -passes=replace-with-veclib -vector-library=AMDLIBM -S %s | FileCheck %s
; A 512-bit external call must not have its operands split by legalization.
target triple = "x86_64-unknown-linux-gnu"

define <8 x double> @generic(<8 x double> %x) #0 {
; CHECK-LABEL: define <8 x double> @generic(
; CHECK-NEXT: [[R:%.*]] = call fast <8 x double> @llvm.log.v8f64(<8 x double> %x)
; CHECK-NEXT: ret <8 x double> [[R]]
  %r = call fast <8 x double> @llvm.log.v8f64(<8 x double> %x)
  ret <8 x double> %r
}

define <8 x double> @avx2(<8 x double> %x) #2 {
; CHECK-LABEL: define <8 x double> @avx2(
; CHECK-NEXT: [[R:%.*]] = call fast <8 x double> @llvm.log.v8f64(<8 x double> %x)
; CHECK-NEXT: ret <8 x double> [[R]]
  %r = call fast <8 x double> @llvm.log.v8f64(<8 x double> %x)
  ret <8 x double> %r
}

define <8 x double> @prefer256(<8 x double> %x) #3 {
; CHECK-LABEL: define <8 x double> @prefer256(
; CHECK-NEXT: [[R:%.*]] = call fast <8 x double> @llvm.log.v8f64(<8 x double> %x)
; CHECK-NEXT: ret <8 x double> [[R]]
  %r = call fast <8 x double> @llvm.log.v8f64(<8 x double> %x)
  ret <8 x double> %r
}

define <8 x double> @avx512(<8 x double> %x) #4 {
; CHECK-LABEL: define <8 x double> @avx512(
; CHECK-NEXT: [[R:%.*]] = call fast <8 x double> @amd_vrd8_log(<8 x double> %x)
; CHECK-NEXT: ret <8 x double> [[R]]
  %r = call fast <8 x double> @llvm.log.v8f64(<8 x double> %x)
  ret <8 x double> %r
}

define <8 x double> @prefer256_no_min(<8 x double> %x) #7 {
; CHECK-LABEL: define <8 x double> @prefer256_no_min(
; CHECK-NEXT: [[R:%.*]] = call fast <8 x double> @amd_vrd8_log(<8 x double> %x)
; CHECK-NEXT: ret <8 x double> [[R]]
  %r = call fast <8 x double> @llvm.log.v8f64(<8 x double> %x)
  ret <8 x double> %r
}

define <8 x double> @prefer256_min512(<8 x double> %x) #8 {
; CHECK-LABEL: define <8 x double> @prefer256_min512(
; CHECK-NEXT: [[R:%.*]] = call fast <8 x double> @amd_vrd8_log(<8 x double> %x)
; CHECK-NEXT: ret <8 x double> [[R]]
  %r = call fast <8 x double> @llvm.log.v8f64(<8 x double> %x)
  ret <8 x double> %r
}

declare <8 x double> @llvm.log.v8f64(<8 x double>)
attributes #0 = { "target-cpu"="x86-64" }
attributes #2 = { "target-cpu"="haswell" }
attributes #3 = { "target-cpu"="skylake-avx512" "prefer-vector-width"="256" "min-legal-vector-width"="256" }
attributes #4 = { "target-cpu"="skylake-avx512" "prefer-vector-width"="512" }

attributes #7 = { "target-cpu"="skylake-avx512" "prefer-vector-width"="256" }
attributes #8 = { "target-cpu"="skylake-avx512" "prefer-vector-width"="256" "min-legal-vector-width"="512" }
