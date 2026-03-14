; RUN: llc < %s -mtriple=ve | FileCheck %s

;;; Test ‘llvm.pow.*’ intrinsic
;;;
;;; Syntax:
;;;   This is an overloaded intrinsic. You can use llvm.pow on any
;;;   floating-point or vector of floating-point type. Not all targets
;;;   support all types however.
;;;
;;; declare float     @llvm.pow.f32(float  %Val, float %Power)
;;; declare double    @llvm.pow.f64(double %Val, double %Power)
;;; declare x86_fp80  @llvm.pow.f80(x86_fp80  %Val, x86_fp80 %Power)
;;; declare fp128     @llvm.pow.f128(fp128 %Val, fp128 %Power)
;;; declare ppc_fp128 @llvm.pow.ppcf128(ppc_fp128  %Val, ppc_fp128 Power)
;;;
;;; Overview:
;;;   The ‘llvm.pow.*’ intrinsics return the first operand raised to
;;;   the specified (positive or negative) power.
;;;
;;; Arguments:
;;;   The arguments and return value are floating-point numbers of
;;;   the same type.
;;;
;;; Note:
;;;   We test only float/double/fp128.

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define float @func_fp_pow_var_float(float noundef %0, float noundef %1) {
; CHECK-LABEL: func_fp_pow_var_float:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s2, powf@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s12, powf@hi(, %s2)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = tail call fast float @llvm.pow.f32(float %0, float %1)
  ret float %3
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn
declare float @llvm.pow.f32(float, float)

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define double @func_fp_pow_var_double(double noundef %0, double noundef %1) {
; CHECK-LABEL: func_fp_pow_var_double:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s2, pow@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s12, pow@hi(, %s2)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = tail call fast double @llvm.pow.f64(double %0, double %1)
  ret double %3
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn
declare double @llvm.pow.f64(double, double)

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define fp128 @func_fp_pow_var_quad(fp128 noundef %0, fp128 noundef %1) {
; CHECK-LABEL: func_fp_pow_var_quad:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s4, powl@lo
; CHECK-NEXT:    and %s4, %s4, (32)0
; CHECK-NEXT:    lea.sl %s12, powl@hi(, %s4)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = tail call fast fp128 @llvm.pow.f128(fp128 %0, fp128 %1)
  ret fp128 %3
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn
declare fp128 @llvm.pow.f128(fp128, fp128)

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn
define float @func_fp_pow_zero_back_float(float noundef %0) {
; CHECK-LABEL: func_fp_pow_zero_back_float:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea.sl %s0, 1065353216
; CHECK-NEXT:    b.l.t (, %s10)
  ret float 1.000000e+00
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn
define double @func_fp_pow_zero_back_double(double noundef %0) {
; CHECK-LABEL: func_fp_pow_zero_back_double:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea.sl %s0, 1072693248
; CHECK-NEXT:    b.l.t (, %s10)
  ret double 1.000000e+00
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn
define fp128 @func_fp_pow_zero_back_quad(fp128 noundef %0) {
; CHECK-LABEL: func_fp_pow_zero_back_quad:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s2, .LCPI{{[0-9]+}}_0@hi(, %s0)
; CHECK-NEXT:    ld %s0, 8(, %s2)
; CHECK-NEXT:    ld %s1, (, %s2)
; CHECK-NEXT:    b.l.t (, %s10)
  ret fp128 0xL00000000000000003FFF000000000000
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define float @func_fp_pow_zero_fore_float(float noundef %0) {
; CHECK-LABEL: func_fp_pow_zero_fore_float:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 0, %s0
; CHECK-NEXT:    lea %s0, powf@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, powf@hi(, %s0)
; CHECK-NEXT:    lea.sl %s0, 0
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = tail call fast float @llvm.pow.f32(float 0.000000e+00, float %0)
  ret float %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define double @func_fp_pow_zero_fore_double(double noundef %0) {
; CHECK-LABEL: func_fp_pow_zero_fore_double:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 0, %s0
; CHECK-NEXT:    lea %s0, pow@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, pow@hi(, %s0)
; CHECK-NEXT:    lea.sl %s0, 0
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = tail call fast double @llvm.pow.f64(double 0.000000e+00, double %0)
  ret double %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define fp128 @func_fp_pow_zero_fore_quad(fp128 noundef %0) {
; CHECK-LABEL: func_fp_pow_zero_fore_quad:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s2, 0, %s0
; CHECK-NEXT:    or %s3, 0, %s1
; CHECK-NEXT:    lea %s0, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s4, .LCPI{{[0-9]+}}_0@hi(, %s0)
; CHECK-NEXT:    ld %s0, 8(, %s4)
; CHECK-NEXT:    ld %s1, (, %s4)
; CHECK-NEXT:    lea %s4, powl@lo
; CHECK-NEXT:    and %s4, %s4, (32)0
; CHECK-NEXT:    lea.sl %s12, powl@hi(, %s4)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = tail call fast fp128 @llvm.pow.f128(fp128 0xL00000000000000000000000000000000, fp128 %0)
  ret fp128 %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define float @func_fp_pow_const_back_float(float noundef %0) {
; CHECK-LABEL: func_fp_pow_const_back_float:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fmul.s %s0, %s0, %s0
; CHECK-NEXT:    lea.sl %s1, 1065353216
; CHECK-NEXT:    fdiv.s %s0, %s1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast float @llvm.powi.f32.i32(float %0, i32 -2)
  ret float %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define double @func_fp_pow_const_back_double(double noundef %0) {
; CHECK-LABEL: func_fp_pow_const_back_double:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fmul.d %s0, %s0, %s0
; CHECK-NEXT:    lea.sl %s1, 1072693248
; CHECK-NEXT:    fdiv.d %s0, %s1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast double @llvm.powi.f64.i32(double %0, i32 -2)
  ret double %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define fp128 @func_fp_pow_const_back_quad(fp128 noundef %0) {
; CHECK-LABEL: func_fp_pow_const_back_quad:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fmul.q %s2, %s0, %s0
; CHECK-NEXT:    lea %s0, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s4, .LCPI{{[0-9]+}}_0@hi(, %s0)
; CHECK-NEXT:    ld %s0, 8(, %s4)
; CHECK-NEXT:    ld %s1, (, %s4)
; CHECK-NEXT:    lea %s4, __divtf3@lo
; CHECK-NEXT:    and %s4, %s4, (32)0
; CHECK-NEXT:    lea.sl %s12, __divtf3@hi(, %s4)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = tail call fast fp128 @llvm.powi.f128.i32(fp128 %0, i32 -2)
  ret fp128 %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define float @func_fp_pow_const_fore_float(float noundef %0) {
; CHECK-LABEL: func_fp_pow_const_fore_float:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 0, %s0
; CHECK-NEXT:    lea %s0, powf@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, powf@hi(, %s0)
; CHECK-NEXT:    lea.sl %s0, -1073741824
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = tail call fast float @llvm.pow.f32(float -2.000000e+00, float %0)
  ret float %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define double @func_fp_pow_const_fore_double(double noundef %0) {
; CHECK-LABEL: func_fp_pow_const_fore_double:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 0, %s0
; CHECK-NEXT:    lea %s0, pow@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, pow@hi(, %s0)
; CHECK-NEXT:    lea.sl %s0, -1073741824
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = tail call fast double @llvm.pow.f64(double -2.000000e+00, double %0)
  ret double %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define fp128 @func_fp_pow_const_fore_quad(fp128 noundef %0) {
; CHECK-LABEL: func_fp_pow_const_fore_quad:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s2, 0, %s0
; CHECK-NEXT:    or %s3, 0, %s1
; CHECK-NEXT:    lea %s0, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s4, .LCPI{{[0-9]+}}_0@hi(, %s0)
; CHECK-NEXT:    ld %s0, 8(, %s4)
; CHECK-NEXT:    ld %s1, (, %s4)
; CHECK-NEXT:    lea %s4, powl@lo
; CHECK-NEXT:    and %s4, %s4, (32)0
; CHECK-NEXT:    lea.sl %s12, powl@hi(, %s4)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = tail call fast fp128 @llvm.pow.f128(fp128 0xL0000000000000000C000000000000000, fp128 %0)
  ret fp128 %2
}

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare float @llvm.powi.f32.i32(float, i32)

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare double @llvm.powi.f64.i32(double, i32)

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare fp128 @llvm.powi.f128.i32(fp128, i32)
