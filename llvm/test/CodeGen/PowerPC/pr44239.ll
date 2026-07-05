; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -verify-machineinstrs < %s | FileCheck %s

define <4 x float> @check_vcfsx(<4 x i32> %a) {
entry:
  %0 = tail call <4 x float> @llvm.ppc.altivec.vcfsx(<4 x i32> %a, i32 1)
  ret  <4 x float> %0
; CHECK-LABEL: check_vcfsx
; CHECK: vcfsx {{[0-9]+}}, {{[0-9]+}}, 1
}

define <4 x float> @check_vcfux(<4 x i32> %a) {
entry:
  %0 = tail call <4 x float> @llvm.ppc.altivec.vcfux(<4 x i32> %a, i32 1)
  ret  <4 x float> %0
; CHECK-LABEL: check_vcfux
; CHECK: vcfux {{[0-9]+}}, {{[0-9]+}}, 1
}

define <4 x i32> @check_vctsxs(<4 x float> %a) {
entry:
  %0 = tail call <4 x i32> @llvm.ppc.altivec.vctsxs(<4 x float> %a, i32 1)
  ret  <4 x i32> %0
; CHECK-LABEL: check_vctsxs
; CHECK: vctsxs {{[0-9]+}}, {{[0-9]+}}, 1
}

define <4 x i32> @check_vctuxs(<4 x float> %a) {
entry:
  %0 = tail call <4 x i32> @llvm.ppc.altivec.vctuxs(<4 x float> %a, i32 1)
  ret  <4 x i32> %0
; CHECK-LABEL: check_vctuxs
; CHECK: vctuxs {{[0-9]+}}, {{[0-9]+}}, 1
}

declare <4 x float> @llvm.ppc.altivec.vcfsx(<4 x i32>, i32 immarg)
declare <4 x float> @llvm.ppc.altivec.vcfux(<4 x i32>, i32 immarg)
declare <4 x i32> @llvm.ppc.altivec.vctsxs(<4 x float>, i32 immarg)
declare <4 x i32> @llvm.ppc.altivec.vctuxs(<4 x float>, i32 immarg)

