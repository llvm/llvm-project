; RUN: llc < %s -mtriple=ve | FileCheck %s

;;; Test ‘llvm.minnum.*’ intrinsic
;;;
;;; Syntax:
;;;   This is an overloaded intrinsic. You can use llvm.minnum on any
;;;   floating-point or vector of floating-point type. Not all targets
;;;   support all types however.
;;;
;;; declare float     @llvm.minnum.f32(float %Val0, float %Val1)
;;; declare double    @llvm.minnum.f64(double %Val0, double %Val1)
;;; declare x86_fp80  @llvm.minnum.f80(x86_fp80 %Val0, x86_fp80 %Val1)
;;; declare fp128     @llvm.minnum.f128(fp128 %Val0, fp128 %Val1)
;;; declare ppc_fp128 @llvm.minnum.ppcf128(ppc_fp128 %Val0, ppc_fp128 %Val1)
;;;
;;; Overview:
;;;   The ‘llvm.minnum.*’ intrinsics return the minimum of the two arguments.
;;;
;;; Arguments:
;;;   The arguments and return value are floating-point numbers of the same
;;;   type.
;;;
;;; Semantics:
;;;   Follows the IEEE-754 semantics for minNum, except for handling of
;;;   signaling NaNs. This match’s the behavior of libm’s fmin.
;;;
;;;   If either operand is a NaN, returns the other non-NaN operand.
;;;   Returns NaN only if both operands are NaN. The returned NaN is
;;;   always quiet. If the operands compare equal, returns a value
;;;   that compares equal to both operands. This means that
;;;   fmin(+/-0.0, +/-0.0) could return either -0.0 or 0.0.
;;;
;;;   Unlike the IEEE-754 2008 behavior, this does not distinguish between
;;;   signaling and quiet NaN inputs. If a target’s implementation follows
;;;   the standard and returns a quiet NaN if either input is a signaling
;;;   NaN, the intrinsic lowering is responsible for quieting the inputs
;;;   to correctly return the non-NaN input (e.g. by using the equivalent
;;;   of llvm.canonicalize).
;;;
;;; Note:
;;;   We test only float/double/fp128.

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define float @func_fp_fmin_var_float(float noundef %0, float noundef %1) {
; CHECK-LABEL: func_fp_fmin_var_float:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fmin.s %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast float @llvm.minnum.f32(float %0, float %1)
  ret float %3
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn
declare float @llvm.minnum.f32(float, float)

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define double @func_fp_fmin_var_double(double noundef %0, double noundef %1) {
; CHECK-LABEL: func_fp_fmin_var_double:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fmin.d %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast double @llvm.minnum.f64(double %0, double %1)
  ret double %3
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn
declare double @llvm.minnum.f64(double, double)

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define fp128 @func_fp_fmin_var_quad(fp128 noundef %0, fp128 noundef %1) {
; CHECK-LABEL: func_fp_fmin_var_quad:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fcmp.q %s4, %s0, %s2
; CHECK-NEXT:    cmov.d.lt %s2, %s0, %s4
; CHECK-NEXT:    cmov.d.lt %s3, %s1, %s4
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    or %s1, 0, %s3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast fp128 @llvm.minnum.f128(fp128 %0, fp128 %1)
  ret fp128 %3
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn
declare fp128 @llvm.minnum.f128(fp128, fp128)

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define float @func_fp_fmin_zero_float(float noundef %0) {
; CHECK-LABEL: func_fp_fmin_zero_float:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fmin.s %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast float @llvm.minnum.f32(float %0, float 0.000000e+00)
  ret float %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define double @func_fp_fmin_zero_double(double noundef %0) {
; CHECK-LABEL: func_fp_fmin_zero_double:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fmin.d %s0, %s0, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast double @llvm.minnum.f64(double %0, double 0.000000e+00)
  ret double %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define fp128 @func_fp_fmin_zero_quad(fp128 noundef %0) {
; CHECK-LABEL: func_fp_fmin_zero_quad:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s4, .LCPI{{[0-9]+}}_0@hi(, %s2)
; CHECK-NEXT:    ld %s2, 8(, %s4)
; CHECK-NEXT:    ld %s3, (, %s4)
; CHECK-NEXT:    fcmp.q %s4, %s0, %s2
; CHECK-NEXT:    cmov.d.lt %s2, %s0, %s4
; CHECK-NEXT:    cmov.d.lt %s3, %s1, %s4
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    or %s1, 0, %s3
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast fp128 @llvm.minnum.f128(fp128 %0, fp128 0xL00000000000000000000000000000000)
  ret fp128 %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define float @func_fp_fmin_const_float(float noundef %0) {
; CHECK-LABEL: func_fp_fmin_const_float:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fmin.s %s0, %s0, (2)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast float @llvm.minnum.f32(float %0, float -2.000000e+00)
  ret float %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define double @func_fp_fmin_const_double(double noundef %0) {
; CHECK-LABEL: func_fp_fmin_const_double:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fmin.d %s0, %s0, (2)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast double @llvm.minnum.f64(double %0, double -2.000000e+00)
  ret double %2
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
define fp128 @func_fp_fmin_const_quad(fp128 noundef %0) {
; CHECK-LABEL: func_fp_fmin_const_quad:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s4, .LCPI{{[0-9]+}}_0@hi(, %s2)
; CHECK-NEXT:    ld %s2, 8(, %s4)
; CHECK-NEXT:    ld %s3, (, %s4)
; CHECK-NEXT:    fcmp.q %s4, %s0, %s2
; CHECK-NEXT:    cmov.d.lt %s2, %s0, %s4
; CHECK-NEXT:    cmov.d.lt %s3, %s1, %s4
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    or %s1, 0, %s3
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast fp128 @llvm.minnum.f128(fp128 %0, fp128 0xL0000000000000000C000000000000000)
  ret fp128 %2
}
