; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=+vsx -O0 -fast-isel=0 -mtriple=powerpc64-unknown-linux-gnu < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

@vf = global <4 x float> <float -1.500000e+00, float 2.500000e+00, float -3.500000e+00, float 4.500000e+00>, align 16
@vd = global <2 x double> <double 3.500000e+00, double -7.500000e+00>, align 16
@d = global double 2.340000e+01, align 8
@vf1 = common global <4 x float> zeroinitializer, align 16
@vd1 = common global <2 x double> zeroinitializer, align 16
@vf2 = common global <4 x float> zeroinitializer, align 16
@vf3 = common global <4 x float> zeroinitializer, align 16
@vd2 = common global <2 x double> zeroinitializer, align 16
@vf4 = common global <4 x float> zeroinitializer, align 16
@d1 = common global double 0.000000e+00, align 8
@d2 = common global double 0.000000e+00, align 8

; Function Attrs: nounwind
define void @test1() #0 {
; CHECK-LABEL: @test1
entry:
  %0 = load volatile <4 x float>, ptr @vf, align 16
  %1 = load volatile <4 x float>, ptr @vf, align 16
  %2 = tail call <4 x float> @llvm.ppc.vsx.xvmaxsp(<4 x float> %0, <4 x float> %1)
; CHECK: xvmaxsp
  store <4 x float> %2, ptr @vf1, align 16
  %3 = load <2 x double>, ptr @vd, align 16
  %4 = tail call <2 x double> @llvm.ppc.vsx.xvmaxdp(<2 x double> %3, <2 x double> %3)
; CHECK: xvmaxdp
  store <2 x double> %4, ptr @vd1, align 16
  %5 = load volatile <4 x float>, ptr @vf, align 16
  %6 = load volatile <4 x float>, ptr @vf, align 16
  %7 = tail call <4 x float> @llvm.ppc.vsx.xvmaxsp(<4 x float> %5, <4 x float> %6)
; CHECK: xvmaxsp
  store <4 x float> %7, ptr @vf2, align 16
  %8 = load volatile <4 x float>, ptr @vf, align 16
  %9 = load volatile <4 x float>, ptr @vf, align 16
  %10 = tail call <4 x float> @llvm.ppc.vsx.xvminsp(<4 x float> %8, <4 x float> %9)
; CHECK: xvminsp
  store <4 x float> %10, ptr @vf3, align 16
  %11 = load <2 x double>, ptr @vd, align 16
  %12 = tail call <2 x double> @llvm.ppc.vsx.xvmindp(<2 x double> %11, <2 x double> %11)
; CHECK: xvmindp
  store <2 x double> %12, ptr @vd2, align 16
  %13 = load volatile <4 x float>, ptr @vf, align 16
  %14 = load volatile <4 x float>, ptr @vf, align 16
  %15 = tail call <4 x float> @llvm.ppc.vsx.xvminsp(<4 x float> %13, <4 x float> %14)
; CHECK: xvminsp
  store <4 x float> %15, ptr @vf4, align 16
  %16 = load double, ptr @d, align 8
  %17 = tail call double @llvm.ppc.vsx.xsmaxdp(double %16, double %16)
; CHECK: xsmaxdp
  store double %17, ptr @d1, align 8
  %18 = tail call double @llvm.ppc.vsx.xsmindp(double %16, double %16)
; CHECK: xsmindp
  store double %18, ptr @d2, align 8
  ret void
}

; Function Attrs: nounwind readnone
declare double @llvm.ppc.vsx.xsmaxdp(double, double)

; Function Attrs: nounwind readnone
declare double @llvm.ppc.vsx.xsmindp(double, double)

; Function Attrs: nounwind readnone
declare <4 x float> @llvm.ppc.vsx.xvminsp(<4 x float>, <4 x float>)

; Function Attrs: nounwind readnone
declare <2 x double> @llvm.ppc.vsx.xvmindp(<2 x double>, <2 x double>)

; Function Attrs: nounwind readnone
declare <4 x float> @llvm.ppc.vsx.xvmaxsp(<4 x float>, <4 x float>)

; Function Attrs: nounwind readnone
declare <2 x double> @llvm.ppc.vsx.xvmaxdp(<2 x double>, <2 x double>)

; Generated from C source:

; % clang -O1 -maltivec -mvsx -S -emit-llvm vsx-minmax.c
;
;volatile vector float vf = { -1.5, 2.5, -3.5, 4.5 };
;vector double vd = { 3.5, -7.5 };
;double d = 23.4;
;
;vector float vf1, vf2, vf3, vf4;
;vector double vd1, vd2;
;double d1, d2;
;
;void test1() {
;  vf1 = vec_max(vf, vf);
;  vd1 = vec_max(vd, vd);
;  vf2 = vec_vmaxfp(vf, vf);
;  vf3 = vec_min(vf, vf);
;  vd2 = vec_min(vd, vd);
;  vf4 = vec_vminfp(vf, vf);
;  d1 = __builtin_vsx_xsmaxdp(d, d);
;  d2 = __builtin_vsx_xsmindp(d, d);
;}
