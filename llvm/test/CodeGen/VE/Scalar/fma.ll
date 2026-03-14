; RUN: llc < %s -mtriple=ve | FileCheck %s

;;; Test ‘llvm.fma.*’ intrinsic
;;;
;;; Syntax:
;;;   This is an overloaded intrinsic. You can use llvm.fma on any
;;;   floating-point or vector of floating-point type. Not all targets
;;;   support all types however.
;;;
;;; declare float     @llvm.fma.f32(float  %a, float  %b, float  %c)
;;; declare double    @llvm.fma.f64(double %a, double %b, double %c)
;;; declare x86_fp80  @llvm.fma.f80(x86_fp80 %a, x86_fp80 %b, x86_fp80 %c)
;;; declare fp128     @llvm.fma.f128(fp128 %a, fp128 %b, fp128 %c)
;;; declare ppc_fp128 @llvm.fma.ppcf128(ppc_fp128 %a, ppc_fp128 %b,
;;;                                     ppc_fp128 %c)
;;;
;;; Overview:
;;;   The ‘llvm.fma.*’ intrinsics perform the fused multiply-add operation.
;;;
;;; Arguments:
;;;   The arguments and return value are floating-point numbers of the same
;;;   type.
;;;
;;; Semantics:
;;;   Return the same value as a corresponding libm ‘fma’ function but without
;;;   trapping or setting errno.
;;;
;;;   When specified with the fast-math-flag ‘afn’, the result may be
;;;   approximated using a less accurate calculation.
;;;
;;; Note:
;;;   We test only float/double/fp128.

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define float @fma_float_var(float noundef %0, float noundef %1, float noundef %2) {
; CHECK-LABEL: fma_float_var:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s3, fmaf@lo
; CHECK-NEXT:    and %s3, %s3, (32)0
; CHECK-NEXT:    lea.sl %s12, fmaf@hi(, %s3)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %4 = tail call fast float @llvm.fma.f32(float %0, float %1, float %2)
  ret float %4
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn
declare float @llvm.fma.f32(float, float, float)

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define double @fma_double_var(double noundef %0, double noundef %1, double noundef %2) {
; CHECK-LABEL: fma_double_var:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s3, fma@lo
; CHECK-NEXT:    and %s3, %s3, (32)0
; CHECK-NEXT:    lea.sl %s12, fma@hi(, %s3)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %4 = tail call fast double @llvm.fma.f64(double %0, double %1, double %2)
  ret double %4
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn
declare double @llvm.fma.f64(double, double, double)

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define fp128 @fma_quad_var(fp128 noundef %0, fp128 noundef %1, fp128 noundef %2) {
; CHECK-LABEL: fma_quad_var:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s6, fmal@lo
; CHECK-NEXT:    and %s6, %s6, (32)0
; CHECK-NEXT:    lea.sl %s12, fmal@hi(, %s6)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %4 = tail call fast fp128 @llvm.fma.f128(fp128 %0, fp128 %1, fp128 %2)
  ret fp128 %4
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn
declare fp128 @llvm.fma.f128(fp128, fp128, fp128)

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn
define float @fma_float_fore_zero(float noundef %0, float noundef returned %1) {
; CHECK-LABEL: fma_float_fore_zero:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  ret float %1
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn
define double @fma_double_fore_zero(double noundef %0, double noundef returned %1) {
; CHECK-LABEL: fma_double_fore_zero:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  ret double %1
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn
define fp128 @fma_quad_fore_zero(fp128 noundef %0, fp128 noundef returned %1) {
; CHECK-LABEL: fma_quad_fore_zero:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    or %s1, 0, %s3
; CHECK-NEXT:    b.l.t (, %s10)
  ret fp128 %1
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn
define float @fma_float_back_zero(float noundef %0, float noundef returned %1) {
; CHECK-LABEL: fma_float_back_zero:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  ret float %1
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn
define double @fma_double_back_zero(double noundef %0, double noundef returned %1) {
; CHECK-LABEL: fma_double_back_zero:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  ret double %1
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn
define fp128 @fma_quad_back_zero(fp128 noundef %0, fp128 noundef returned %1) {
; CHECK-LABEL: fma_quad_back_zero:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    or %s1, 0, %s3
; CHECK-NEXT:    b.l.t (, %s10)
  ret fp128 %1
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define float @fma_float_fore_const(float noundef %0, float noundef %1) {
; CHECK-LABEL: fma_float_fore_const:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s2, 0, %s1
; CHECK-NEXT:    lea %s1, fmaf@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s12, fmaf@hi(, %s1)
; CHECK-NEXT:    lea.sl %s1, -1073741824
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = tail call fast float @llvm.fma.f32(float %0, float -2.000000e+00, float %1)
  ret float %3
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define double @fma_double_fore_const(double noundef %0, double noundef %1) {
; CHECK-LABEL: fma_double_fore_const:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s2, 0, %s1
; CHECK-NEXT:    lea %s1, fma@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s12, fma@hi(, %s1)
; CHECK-NEXT:    lea.sl %s1, -1073741824
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = tail call fast double @llvm.fma.f64(double %0, double -2.000000e+00, double %1)
  ret double %3
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define fp128 @fma_quad_fore_const(fp128 noundef %0, fp128 noundef %1) {
; CHECK-LABEL: fma_quad_fore_const:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s4, 0, %s2
; CHECK-NEXT:    or %s5, 0, %s3
; CHECK-NEXT:    lea %s2, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s6, .LCPI{{[0-9]+}}_0@hi(, %s2)
; CHECK-NEXT:    ld %s2, 8(, %s6)
; CHECK-NEXT:    ld %s3, (, %s6)
; CHECK-NEXT:    lea %s6, fmal@lo
; CHECK-NEXT:    and %s6, %s6, (32)0
; CHECK-NEXT:    lea.sl %s12, fmal@hi(, %s6)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = tail call fast fp128 @llvm.fma.f128(fp128 %0, fp128 0xL0000000000000000C000000000000000, fp128 %1)
  ret fp128 %3
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define float @fma_float_back_const(float noundef %0, float noundef %1) {
; CHECK-LABEL: fma_float_back_const:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s2, 0, %s1
; CHECK-NEXT:    lea %s1, fmaf@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s12, fmaf@hi(, %s1)
; CHECK-NEXT:    lea.sl %s1, -1073741824
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = tail call fast float @llvm.fma.f32(float %0, float -2.000000e+00, float %1)
  ret float %3
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define double @fma_double_back_const(double noundef %0, double noundef %1) {
; CHECK-LABEL: fma_double_back_const:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s2, 0, %s1
; CHECK-NEXT:    lea %s1, fma@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s12, fma@hi(, %s1)
; CHECK-NEXT:    lea.sl %s1, -1073741824
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = tail call fast double @llvm.fma.f64(double %0, double -2.000000e+00, double %1)
  ret double %3
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define fp128 @fma_quad_back_const(fp128 noundef %0, fp128 noundef %1) {
; CHECK-LABEL: fma_quad_back_const:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s4, 0, %s2
; CHECK-NEXT:    or %s5, 0, %s3
; CHECK-NEXT:    lea %s2, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s6, .LCPI{{[0-9]+}}_0@hi(, %s2)
; CHECK-NEXT:    ld %s2, 8(, %s6)
; CHECK-NEXT:    ld %s3, (, %s6)
; CHECK-NEXT:    lea %s6, fmal@lo
; CHECK-NEXT:    and %s6, %s6, (32)0
; CHECK-NEXT:    lea.sl %s12, fmal@hi(, %s6)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = tail call fast fp128 @llvm.fma.f128(fp128 %0, fp128 0xL0000000000000000C000000000000000, fp128 %1)
  ret fp128 %3
}
