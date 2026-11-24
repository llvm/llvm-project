; RUN: llc < %s -mtriple=armv7-eabi -mattr=+neon,+vfp4 | FileCheck %s
; RUN: llc < %s -mtriple=arm-arm-eabi -mcpu=cortex-m7  | FileCheck %s
; RUN: llc < %s -mtriple=arm-arm-eabi -mcpu=cortex-m4  | FileCheck %s -check-prefix=DONT-FUSE
; RUN: llc < %s -mtriple=arm-arm-eabi -mcpu=cortex-m33 | FileCheck %s -check-prefix=DONT-FUSE

; Check generated fused MAC and MLS.

define arm_aapcs_vfpcc double @fusedMACTest1(double %d1, double %d2, double %d3) {
;CHECK-LABEL: fusedMACTest1:
;CHECK: vfma.f64
  %1 = fmul contract double %d1, %d2
  %2 = fadd contract double %1, %d3
  ret double %2
}

define arm_aapcs_vfpcc float @fusedMACTest2(float %f1, float %f2, float %f3) {
;CHECK-LABEL: fusedMACTest2:
;CHECK: vfma.f32

;DONT-FUSE-LABEL: fusedMACTest2:
;DONT-FUSE:       vmul.f32
;DONT-FUSE-NEXT:  vadd.f32

  %1 = fmul contract float %f1, %f2
  %2 = fadd contract float %1, %f3
  ret float %2
}

define arm_aapcs_vfpcc double @fusedMACTest3(double %d1, double %d2, double %d3) {
;CHECK-LABEL: fusedMACTest3:
;CHECK: vfms.f64
  %1 = fmul contract double %d2, %d3
  %2 = fsub contract double %d1, %1
  ret double %2
}

define arm_aapcs_vfpcc float @fusedMACTest4(float %f1, float %f2, float %f3) {
;CHECK-LABEL: fusedMACTest4:
;CHECK: vfms.f32
  %1 = fmul contract float %f2, %f3
  %2 = fsub contract float %f1, %1
  ret float %2
}

define arm_aapcs_vfpcc double @fusedMACTest5(double %d1, double %d2, double %d3) {
;CHECK-LABEL: fusedMACTest5:
;CHECK: vfnma.f64
  %1 = fmul contract double %d1, %d2
  %2 = fsub double -0.0, %1
  %3 = fsub contract double %2, %d3
  ret double %3
}

define arm_aapcs_vfpcc float @fusedMACTest6(float %f1, float %f2, float %f3) {
;CHECK-LABEL: fusedMACTest6:
;CHECK: vfnma.f32
  %1 = fmul contract float %f1, %f2
  %2 = fsub float -0.0, %1
  %3 = fsub contract float %2, %f3
  ret float %3
}

define arm_aapcs_vfpcc double @fusedMACTest7(double %d1, double %d2, double %d3) {
;CHECK-LABEL: fusedMACTest7:
;CHECK: vfnms.f64
  %1 = fmul contract double %d1, %d2
  %2 = fsub contract double %1, %d3
  ret double %2
}

define arm_aapcs_vfpcc float @fusedMACTest8(float %f1, float %f2, float %f3) {
;CHECK-LABEL: fusedMACTest8:
;CHECK: vfnms.f32
  %1 = fmul contract float %f1, %f2
  %2 = fsub contract float %1, %f3
  ret float %2
}

define arm_aapcs_vfpcc <2 x float> @fusedMACTest9(<2 x float> %a, <2 x float> %b) {
;CHECK-LABEL: fusedMACTest9:
;CHECK: vfma.f32
  %mul = fmul contract <2 x float> %a, %b
  %add = fadd contract <2 x float> %mul, %a
  ret <2 x float> %add
}

define arm_aapcs_vfpcc <2 x float> @fusedMACTest10(<2 x float> %a, <2 x float> %b) {
;CHECK-LABEL: fusedMACTest10:
;CHECK: vfms.f32
  %mul = fmul contract <2 x float> %a, %b
  %sub = fsub contract <2 x float> %a, %mul
  ret <2 x float> %sub
}

define arm_aapcs_vfpcc <4 x float> @fusedMACTest11(<4 x float> %a, <4 x float> %b) {
;CHECK-LABEL: fusedMACTest11:
;CHECK: vfma.f32
  %mul = fmul contract <4 x float> %a, %b
  %add = fadd contract <4 x float> %mul, %a
  ret <4 x float> %add
}

define arm_aapcs_vfpcc <4 x float> @fusedMACTest12(<4 x float> %a, <4 x float> %b) {
;CHECK-LABEL: fusedMACTest12:
;CHECK: vfms.f32
  %mul = fmul contract <4 x float> %a, %b
  %sub = fsub contract <4 x float> %a, %mul
  ret <4 x float> %sub
}

define arm_aapcs_vfpcc float @test_fma_f32(float %a, float %b, float %c) nounwind readnone ssp {
entry:
; CHECK: test_fma_f32
; CHECK: vfma.f32
  %tmp1 = tail call float @llvm.fma.f32(float %a, float %b, float %c) nounwind readnone
  ret float %tmp1
}

define arm_aapcs_vfpcc double @test_fma_f64(double %a, double %b, double %c) nounwind readnone ssp {
entry:
; CHECK: test_fma_f64
; CHECK: vfma.f64
  %tmp1 = tail call double @llvm.fma.f64(double %a, double %b, double %c) nounwind readnone
  ret double %tmp1
}

define arm_aapcs_vfpcc <2 x float> @test_fma_v2f32(<2 x float> %a, <2 x float> %b, <2 x float> %c) nounwind readnone ssp {
entry:
; CHECK: test_fma_v2f32
; CHECK: vfma.f32
  %tmp1 = tail call <2 x float> @llvm.fma.v2f32(<2 x float> %a, <2 x float> %b, <2 x float> %c) nounwind
  ret <2 x float> %tmp1
}

define arm_aapcs_vfpcc double @test_fms_f64(double %a, double %b, double %c) nounwind readnone ssp {
entry:
; CHECK: test_fms_f64
; CHECK: vfms.f64
  %tmp1 = fsub double -0.0, %a
  %tmp2 = tail call double @llvm.fma.f64(double %tmp1, double %b, double %c) nounwind readnone
  ret double %tmp2
}

define arm_aapcs_vfpcc double @test_fms_f64_2(double %a, double %b, double %c) nounwind readnone ssp {
entry:
; CHECK: test_fms_f64_2
; CHECK: vfms.f64
  %tmp1 = fsub double -0.0, %b
  %tmp2 = tail call double @llvm.fma.f64(double %a, double %tmp1, double %c) nounwind readnone
  ret double %tmp2
}

define arm_aapcs_vfpcc float @test_fnms_f32(float %a, float %b, ptr %c) nounwind readnone ssp {
; CHECK: test_fnms_f32
; CHECK: vfnms.f32
  %tmp1 = load float, ptr %c, align 4
  %tmp2 = fsub float -0.0, %tmp1
  %tmp3 = tail call float @llvm.fma.f32(float %a, float %b, float %tmp2) nounwind readnone
  ret float %tmp3 
}

define arm_aapcs_vfpcc double @test_fnms_f64(double %a, double %b, double %c) nounwind readnone ssp {
entry:
; CHECK: test_fnms_f64
; CHECK: vfnms.f64
  %tmp1 = fsub double -0.0, %a
  %tmp2 = tail call double @llvm.fma.f64(double %tmp1, double %b, double %c) nounwind readnone
  %tmp3 = fsub double -0.0, %tmp2
  ret double %tmp3
}

define arm_aapcs_vfpcc double @test_fnms_f64_2(double %a, double %b, double %c) nounwind readnone ssp {
entry:
; CHECK: test_fnms_f64_2
; CHECK: vfnms.f64
  %tmp1 = fsub double -0.0, %b
  %tmp2 = tail call double @llvm.fma.f64(double %a, double %tmp1, double %c) nounwind readnone
  %tmp3 = fsub double -0.0, %tmp2
  ret double %tmp3
}

define arm_aapcs_vfpcc double @test_fnma_f64(double %a, double %b, double %c) nounwind readnone ssp {
entry:
; CHECK: test_fnma_f64
; CHECK: vfnma.f64
  %tmp1 = tail call double @llvm.fma.f64(double %a, double %b, double %c) nounwind readnone
  %tmp2 = fsub double -0.0, %tmp1
  ret double %tmp2
}

define arm_aapcs_vfpcc double @test_fnma_f64_2(double %a, double %b, double %c) nounwind readnone ssp {
entry:
; CHECK: test_fnma_f64_2
; CHECK: vfnma.f64
  %tmp1 = fsub double -0.0, %a
  %tmp2 = fsub double -0.0, %c
  %tmp3 = tail call double @llvm.fma.f64(double %tmp1, double %b, double %tmp2) nounwind readnone
  ret double %tmp3
}

define arm_aapcs_vfpcc float @test_fma_const_fold(float %a, float %b) nounwind {
; CHECK: test_fma_const_fold
; CHECK-NOT: vfma
; CHECK-NOT: vmul
; CHECK: vadd
  %ret = call float @llvm.fma.f32(float %a, float 1.0, float %b)
  ret float %ret
}

define arm_aapcs_vfpcc float @test_fma_canonicalize(float %a, float %b) nounwind {
; CHECK: test_fma_canonicalize
; CHECK: vmov.f32 [[R1:s[0-9]+]], #2.000000e+00
; CHECK: vfma.f32 {{s[0-9]+}}, {{s[0-9]+}}, [[R1]]
  %ret = call float @llvm.fma.f32(float 2.0, float %a, float %b)
  ret float %ret
}

; Check that very wide vector fma's can be split into legal fma's.
define arm_aapcs_vfpcc void @test_fma_v8f32(<8 x float> %a, <8 x float> %b, <8 x float> %c, ptr %p) nounwind readnone ssp {
; CHECK: test_fma_v8f32
; CHECK: vfma.f32
; CHECK: vfma.f32
entry:
  %call = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %a, <8 x float> %b, <8 x float> %c) nounwind readnone
  store <8 x float> %call, ptr %p, align 16
  ret void
}

; Fusion requires the contract flag on both the multiply and the add/subtract.
; With the flag on only one of them, no fused VFMA/VFMS is formed: cores with
; VMLA/VMLS emit the unfused MAC, others emit a separate vmul and vadd/vsub.

define arm_aapcs_vfpcc float @noFuseAddOnly(float %f1, float %f2, float %f3) {
; CHECK-LABEL: noFuseAddOnly:
; CHECK: vmla.f32
;
; DONT-FUSE-LABEL: noFuseAddOnly:
; DONT-FUSE: vmul.f32
; DONT-FUSE-NEXT: vadd.f32
  %1 = fmul float %f1, %f2
  %2 = fadd contract float %1, %f3
  ret float %2
}

define arm_aapcs_vfpcc float @noFuseMulOnly(float %f1, float %f2, float %f3) {
; CHECK-LABEL: noFuseMulOnly:
; CHECK: vmla.f32
;
; DONT-FUSE-LABEL: noFuseMulOnly:
; DONT-FUSE: vmul.f32
; DONT-FUSE-NEXT: vadd.f32
  %1 = fmul contract float %f1, %f2
  %2 = fadd float %1, %f3
  ret float %2
}

define arm_aapcs_vfpcc float @noFuseSubAddOnly(float %f1, float %f2, float %f3) {
; CHECK-LABEL: noFuseSubAddOnly:
; CHECK: vmls.f32
;
; DONT-FUSE-LABEL: noFuseSubAddOnly:
; DONT-FUSE: vmul.f32
; DONT-FUSE-NEXT: vsub.f32
  %1 = fmul float %f2, %f3
  %2 = fsub contract float %f1, %1
  ret float %2
}

define arm_aapcs_vfpcc float @noFuseSubMulOnly(float %f1, float %f2, float %f3) {
; CHECK-LABEL: noFuseSubMulOnly:
; CHECK: vmls.f32
;
; DONT-FUSE-LABEL: noFuseSubMulOnly:
; DONT-FUSE: vmul.f32
; DONT-FUSE-NEXT: vsub.f32
  %1 = fmul contract float %f2, %f3
  %2 = fsub float %f1, %1
  ret float %2
}

; Vector forms exercise the both-operands-contract requirement on the isel
; patterns (DAGCombiner does not fuse these without MVE). The unfused MAC is
; emitted instead of a VFMA.
define arm_aapcs_vfpcc <4 x float> @noFuseVecAddOnly(<4 x float> %a, <4 x float> %b, <4 x float> %c) {
; CHECK-LABEL: noFuseVecAddOnly:
; CHECK: vmla.f32
;
; DONT-FUSE-LABEL: noFuseVecAddOnly:
; DONT-FUSE: vmul.f32
; DONT-FUSE: vadd.f32
  %m = fmul <4 x float> %a, %b
  %r = fadd contract <4 x float> %m, %c
  ret <4 x float> %r
}

define arm_aapcs_vfpcc <4 x float> @noFuseVecMulOnly(<4 x float> %a, <4 x float> %b, <4 x float> %c) {
; CHECK-LABEL: noFuseVecMulOnly:
; CHECK: vmla.f32
;
; DONT-FUSE-LABEL: noFuseVecMulOnly:
; DONT-FUSE: vmul.f32
; DONT-FUSE: vadd.f32
  %m = fmul contract <4 x float> %a, %b
  %r = fadd <4 x float> %m, %c
  ret <4 x float> %r
}


declare float @llvm.fma.f32(float, float, float) nounwind readnone
declare double @llvm.fma.f64(double, double, double) nounwind readnone
declare <2 x float> @llvm.fma.v2f32(<2 x float>, <2 x float>, <2 x float>) nounwind readnone
declare <8 x float> @llvm.fma.v8f32(<8 x float>, <8 x float>, <8 x float>) nounwind readnone
