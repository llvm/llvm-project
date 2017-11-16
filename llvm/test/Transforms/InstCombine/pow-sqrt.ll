; RUN: opt < %s -instcombine -S | FileCheck %s

define double @pow_half(double %x) {
; CHECK-LABEL: @pow_half(
; CHECK-NEXT:    [[SQRT:%.*]] = call fast double @sqrt(double %x) #1
; CHECK-NEXT:    ret double [[SQRT]]
;
  %pow = call fast double @llvm.pow.f64(double %x, double 5.000000e-01)
  ret double %pow
}

define double @pow_neghalf(double %x) {
; CHECK-LABEL: @pow_neghalf(
; CHECK-NEXT:    [[SQRT:%.*]] = call fast double @sqrt(double %x) #1
; CHECK-NEXT:    [[SQRTRECIP:%.*]] = fdiv fast double 1.000000e+00, [[SQRT]]
; CHECK-NEXT:    ret double [[SQRTRECIP]]
;
  %pow = call fast double @llvm.pow.f64(double %x, double -5.000000e-01)
  ret double %pow
}

define double @pow_half_approx(double %x) {
; CHECK-LABEL: @pow_half_approx(
; CHECK-NEXT:    [[SQRT:%.*]] = call double @sqrt(double %x) #1
; CHECK-NEXT:    [[TMP1:%.*]] = call double @llvm.fabs.f64(double [[SQRT]])
; CHECK-NEXT:    [[TMP2:%.*]] = fcmp oeq double %x, 0xFFF0000000000000
; CHECK-NEXT:    [[TMP3:%.*]] = select i1 [[TMP2]], double 0x7FF0000000000000, double [[TMP1]]
; CHECK-NEXT:    ret double [[TMP3]]
;
  %pow = call afn double @llvm.pow.f64(double %x, double 5.000000e-01)
  ret double %pow
}

define double @pow_neghalf_approx(double %x) {
; CHECK-LABEL: @pow_neghalf_approx(
; CHECK-NEXT:    [[POW:%.*]] = call afn double @llvm.pow.f64(double %x, double -5.000000e-01)
; CHECK-NEXT:    ret double [[POW]]
;
  %pow = call afn double @llvm.pow.f64(double %x, double -5.000000e-01)
  ret double %pow
}

declare double @llvm.pow.f64(double, double) #0

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { nounwind readnone }
