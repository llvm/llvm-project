; RUN: opt < %s -instcombine -S | FileCheck %s

define double @pow_intrinsic_half_fast(double %x) {
; CHECK-LABEL: @pow_intrinsic_half_fast(
; CHECK-NEXT:    [[TMP1:%.*]] = call fast double @llvm.sqrt.f64(double %x)
; CHECK-NEXT:    ret double [[TMP1]]
;
  %pow = call fast double @llvm.pow.f64(double %x, double 5.000000e-01)
  ret double %pow
}

define <2 x double> @pow_intrinsic_half_approx(<2 x double> %x) {
; CHECK-LABEL: @pow_intrinsic_half_approx(
; CHECK-NEXT:    [[POW:%.*]] = call afn <2 x double> @llvm.pow.v2f64(<2 x double> %x, <2 x double> <double 5.000000e-01, double 5.000000e-01>)
; CHECK-NEXT:    ret <2 x double> [[POW]]
;
  %pow = call afn <2 x double> @llvm.pow.v2f64(<2 x double> %x, <2 x double> <double 5.0e-01, double 5.0e-01>)
  ret <2 x double> %pow
}

define double @pow_libcall_half_approx(double %x) {
; CHECK-LABEL: @pow_libcall_half_approx(
; CHECK-NEXT:    [[SQRT:%.*]] = call double @sqrt(double %x)
; CHECK-NEXT:    [[TMP1:%.*]] = call double @llvm.fabs.f64(double [[SQRT]])
; CHECK-NEXT:    [[TMP2:%.*]] = fcmp oeq double %x, 0xFFF0000000000000
; CHECK-NEXT:    [[TMP3:%.*]] = select i1 [[TMP2]], double 0x7FF0000000000000, double [[TMP1]]
; CHECK-NEXT:    ret double [[TMP3]]
;
  %pow = call afn double @pow(double %x, double 5.0e-01)
  ret double %pow
}

define <2 x double> @pow_intrinsic_neghalf_fast(<2 x double> %x) {
; CHECK-LABEL: @pow_intrinsic_neghalf_fast(
; CHECK-NEXT:    [[TMP1:%.*]] = call fast <2 x double> @llvm.sqrt.v2f64(<2 x double> %x)
; CHECK-NEXT:    [[TMP2:%.*]] = fdiv fast <2 x double> <double 1.000000e+00, double 1.000000e+00>, [[TMP1]]
; CHECK-NEXT:    ret <2 x double> [[TMP2]]
;
  %pow = call fast <2 x double> @llvm.pow.v2f64(<2 x double> %x, <2 x double> <double -5.0e-01, double -5.0e-01>)
  ret <2 x double> %pow
}

define double @pow_intrinsic_neghalf_approx(double %x) {
; CHECK-LABEL: @pow_intrinsic_neghalf_approx(
; CHECK-NEXT:    [[POW:%.*]] = call afn double @llvm.pow.f64(double %x, double -5.000000e-01)
; CHECK-NEXT:    ret double [[POW]]
;
  %pow = call afn double @llvm.pow.f64(double %x, double -5.0e-01)
  ret double %pow
}

define float @pow_libcall_neghalf_fast(float %x) {
; CHECK-LABEL: @pow_libcall_neghalf_fast(
; CHECK-NEXT:    [[SQRTF:%.*]] = call fast float @sqrtf(float %x)
; CHECK-NEXT:    [[TMP1:%.*]] = fdiv fast float 1.000000e+00, [[SQRTF]]
; CHECK-NEXT:    ret float [[TMP1]]
;
  %pow = call fast float @powf(float %x, float -5.0e-01)
  ret float %pow
}

declare double @llvm.pow.f64(double, double) #0
declare <2 x double> @llvm.pow.v2f64(<2 x double>, <2 x double>) #0
declare double @pow(double, double)
declare float @powf(float, float)

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { nounwind readnone }

