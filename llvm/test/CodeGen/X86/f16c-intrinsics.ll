; RUN: llc < %s -mtriple=i686-unknown-unknown   -mattr=+avx,+f16c | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx,+f16c | FileCheck %s

define <4 x float> @test_x86_vcvtph2ps_128(<8 x i16> %a0) {
  ; CHECK-LABEL: test_x86_vcvtph2ps_128:
  ; CHECK-NOT: vmov
  ; CHECK: vcvtph2ps
  %res = call <4 x float> @llvm.x86.vcvtph2ps.128(<8 x i16> %a0) ; <<4 x float>> [#uses=1]
  ret <4 x float> %res
}
declare <4 x float> @llvm.x86.vcvtph2ps.128(<8 x i16>) nounwind readonly


define <8 x float> @test_x86_vcvtph2ps_256(<8 x i16> %a0) {
  ; CHECK-LABEL: test_x86_vcvtph2ps_256:
  ; CHECK-NOT: vmov
  ; CHECK: vcvtph2ps
  %res = call <8 x float> @llvm.x86.vcvtph2ps.256(<8 x i16> %a0) ; <<8 x float>> [#uses=1]
  ret <8 x float> %res
}
declare <8 x float> @llvm.x86.vcvtph2ps.256(<8 x i16>) nounwind readonly

define <8 x float> @test_x86_vcvtph2ps_256_m(<8 x i16>* nocapture %a) nounwind {
entry:
  ; CHECK-LABEL: test_x86_vcvtph2ps_256_m:
  ; CHECK-NOT: vmov
  ; CHECK: vcvtph2ps  (%
  %tmp1 = load <8 x i16>, <8 x i16>* %a, align 16
  %0 = tail call <8 x float> @llvm.x86.vcvtph2ps.256(<8 x i16> %tmp1)
  ret <8 x float> %0
}

define <8 x i16> @test_x86_vcvtps2ph_128(<4 x float> %a0) {
  ; CHECK-LABEL: test_x86_vcvtps2ph_128:
  ; CHECK-NOT: vmov
  ; CHECK: vcvtps2ph
  %res = call <8 x i16> @llvm.x86.vcvtps2ph.128(<4 x float> %a0, i32 0) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.vcvtps2ph.128(<4 x float>, i32) nounwind readonly

define <8 x i16> @test_x86_vcvtps2ph_256(<8 x float> %a0) {
  ; CHECK-LABEL: test_x86_vcvtps2ph_256:
  ; CHECK-NOT: vmov
  ; CHECK: vcvtps2ph
  %res = call <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float> %a0, i32 0) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float>, i32) nounwind readonly

define <4 x float> @test_x86_vcvtps2ph_128_scalar(i64* %ptr) {
; CHECK-LABEL: test_x86_vcvtps2ph_128_scalar:
; CHECK-NOT: vmov
; CHECK: vcvtph2ps (%

  %load = load i64, i64* %ptr
  %ins1 = insertelement <2 x i64> undef, i64 %load, i32 0
  %ins2 = insertelement <2 x i64> %ins1, i64 0, i32 1
  %bc = bitcast <2 x i64> %ins2 to <8 x i16>
  %res = tail call <4 x float> @llvm.x86.vcvtph2ps.128(<8 x i16> %bc) #2
  ret <4 x float> %res
}

define void @test_x86_vcvtps2ph_256_m(<8 x i16>* nocapture %d, <8 x float> %a) nounwind {
entry:
  ; CHECK-LABEL: test_x86_vcvtps2ph_256_m:
  ; CHECK-NOT: vmov
  ; CHECK: vcvtps2ph  $3, %ymm0, (%
  %0 = tail call <8 x i16> @llvm.x86.vcvtps2ph.256(<8 x float> %a, i32 3)
  store <8 x i16> %0, <8 x i16>* %d, align 16
  ret void
}

define void @test_x86_vcvtps2ph_128_m(<4 x i16>* nocapture %d, <4 x float> %a) nounwind {
entry:
  ; CHECK-LABEL: test_x86_vcvtps2ph_128_m:
  ; CHECK-NOT: vmov
  ; CHECK: vcvtps2ph  $3, %xmm0, (%
  %0 = tail call <8 x i16> @llvm.x86.vcvtps2ph.128(<4 x float> %a, i32 3)
  %1 = shufflevector <8 x i16> %0, <8 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  store <4 x i16> %1, <4 x i16>* %d, align 8
  ret void
}

define void @test_x86_vcvtps2ph_128_m2(double* nocapture %hf4x16, <4 x float> %f4x32) #0 {
entry:
  ; CHECK-LABEL: test_x86_vcvtps2ph_128_m2:
  ; CHECK-NOT: vmov
  ; CHECK: vcvtps2ph  $3, %xmm0, (%
  %0 = tail call <8 x i16> @llvm.x86.vcvtps2ph.128(<4 x float> %f4x32, i32 3)
  %1 = bitcast <8 x i16> %0 to <2 x double>
  %vecext = extractelement <2 x double> %1, i32 0
  store double %vecext, double* %hf4x16, align 8
  ret void
}

define void @test_x86_vcvtps2ph_128_m3(i64* nocapture %hf4x16, <4 x float> %f4x32) #0 {
entry:
  ; CHECK-LABEL: test_x86_vcvtps2ph_128_m3:
  ; CHECK-NOT: vmov
  ; CHECK: vcvtps2ph  $3, %xmm0, (%
  %0 = tail call <8 x i16> @llvm.x86.vcvtps2ph.128(<4 x float> %f4x32, i32 3)
  %1 = bitcast <8 x i16> %0 to <2 x i64>
  %vecext = extractelement <2 x i64> %1, i32 0
  store i64 %vecext, i64* %hf4x16, align 8
  ret void
}
