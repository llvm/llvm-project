; Test vector intrinsics added with arch14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=arch14 | FileCheck %s

declare <4 x float> @llvm.s390.vclfnhs(<8 x i16>, i32)
declare <4 x float> @llvm.s390.vclfnls(<8 x i16>, i32)
declare <8 x i16> @llvm.s390.vcrnfs(<4 x float>, <4 x float>, i32)
declare <8 x i16> @llvm.s390.vcfn(<8 x i16>, i32)
declare <8 x i16> @llvm.s390.vcnf(<8 x i16>, i32)

; VCLFNH.
define <4 x float> @test_vclfnhs(<8 x i16> %a) {
; CHECK-LABEL: test_vclfnhs:
; CHECK: vclfnh %v24, %v24, 2, 0
; CHECK: br %r14
  %res = call <4 x float> @llvm.s390.vclfnhs(<8 x i16> %a, i32 0)
  ret <4 x float> %res
}

; VCLFNL.
define <4 x float> @test_vclfnls(<8 x i16> %a) {
; CHECK-LABEL: test_vclfnls:
; CHECK: vclfnl %v24, %v24, 2, 0
; CHECK: br %r14
  %res = call <4 x float> @llvm.s390.vclfnls(<8 x i16> %a, i32 0)
  ret <4 x float> %res
}

; VCRNF.
define <8 x i16> @test_vcrnfs(<4 x float> %a, <4 x float> %b) {
; CHECK-LABEL: test_vcrnfs:
; CHECK: vcrnf %v24, %v24, %v26, 0, 2
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vcrnfs(<4 x float> %a, <4 x float> %b, i32 0)
  ret <8 x i16> %res
}

; VCFN.
define <8 x i16> @test_vcfn(<8 x i16> %a) {
; CHECK-LABEL: test_vcfn:
; CHECK: vcfn %v24, %v24, 1, 0
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vcfn(<8 x i16> %a, i32 0)
  ret <8 x i16> %res
}

; VCNF.
define <8 x i16> @test_vcnf(<8 x i16> %a) {
; CHECK-LABEL: test_vcnf:
; CHECK: vcnf %v24, %v24, 0, 1
; CHECK: br %r14
  %res = call <8 x i16> @llvm.s390.vcnf(<8 x i16> %a, i32 0)
  ret <8 x i16> %res
}
