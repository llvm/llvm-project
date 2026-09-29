; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=+vsx -O1 -mtriple=powerpc64-unknown-linux-gnu < %s | FileCheck %s

@vf = global <4 x float> <float -1.500000e+00, float 2.500000e+00, float -3.500000e+00, float 4.500000e+00>, align 16
@vd = global <2 x double> <double 3.500000e+00, double -7.500000e+00>, align 16
@vf_res = common global <4 x float> zeroinitializer, align 16
@vd_res = common global <2 x double> zeroinitializer, align 16

define void @test1() {
entry:
  %0 = load <4 x float>, ptr @vf, align 16
  %1 = tail call <4 x float> @llvm.ppc.vsx.xvdivsp(<4 x float> %0, <4 x float> %0)
  store <4 x float> %1, ptr @vf_res, align 16
  ret void
}
; CHECK-LABEL: @test1
; CHECK: xvdivsp

define void @test2() {
entry:
  %0 = load <2 x double>, ptr @vd, align 16
  %1 = tail call <2 x double> @llvm.ppc.vsx.xvdivdp(<2 x double> %0, <2 x double> %0)
  store <2 x double> %1, ptr @vd_res, align 16
  ret void
}
; CHECK-LABEL: @test2
; CHECK: xvdivdp

declare <2 x double> @llvm.ppc.vsx.xvdivdp(<2 x double>, <2 x double>)
declare <4 x float> @llvm.ppc.vsx.xvdivsp(<4 x float>, <4 x float>)
