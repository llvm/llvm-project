; RUN: llc < %s -mtriple=ve | FileCheck %s

;;; Test ‘llvm.cos.*’ intrinsic
;;;
;;; Syntax:
;;;   This is an overloaded intrinsic. You can use llvm.cos on any
;;;   floating-point or vector of floating-point type. Not all targets
;;;   support all types however.
;;;
;;; declare float     @llvm.cos.f32(float  %Val)
;;; declare double    @llvm.cos.f64(double %Val)
;;; declare x86_fp80  @llvm.cos.f80(x86_fp80  %Val)
;;; declare fp128     @llvm.cos.f128(fp128 %Val)
;;; declare ppc_fp128 @llvm.cos.ppcf128(ppc_fp128  %Val)
;;;
;;; Overview:
;;;   The ‘llvm.cos.*’ intrinsics return the cosine of the operand.
;;;
;;; Arguments:
;;;   The argument and return value are floating-point numbers of the same type.
;;;
;;; Semantics:
;;;   Return the same value as a corresponding libm ‘cos’ function but without
;;;   trapping or setting errno.
;;;
;;;   When specified with the fast-math-flag ‘afn’, the result may be
;;;   approximated using a less accurate calculation.
;;;
;;; Note:
;;;   We test only float/double/fp128.

; Function Attrs: nounwind readnone
define float @fcos_float_var(float %0) {
; CHECK-LABEL: fcos_float_var:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s1, cosf@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s12, cosf@hi(, %s1)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = tail call fast float @llvm.cos.f32(float %0)
  ret float %2
}

; Function Attrs: nounwind readnone speculatable willreturn
declare float @llvm.cos.f32(float)

; Function Attrs: nounwind readnone
define double @fcos_double_var(double %0) {
; CHECK-LABEL: fcos_double_var:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s1, cos@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s12, cos@hi(, %s1)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = tail call fast double @llvm.cos.f64(double %0)
  ret double %2
}

; Function Attrs: nounwind readnone speculatable willreturn
declare double @llvm.cos.f64(double)

; Function Attrs: nounwind readnone
define fp128 @fcos_quad_var(fp128 %0) {
; CHECK-LABEL: fcos_quad_var:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s2, cosl@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s12, cosl@hi(, %s2)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = tail call fast fp128 @llvm.cos.f128(fp128 %0)
  ret fp128 %2
}

; Function Attrs: nounwind readnone speculatable willreturn
declare fp128 @llvm.cos.f128(fp128)

; Function Attrs: norecurse nounwind readnone
define float @fcos_float_zero() {
; CHECK-LABEL: fcos_float_zero:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s0, 1065353216
; CHECK-NEXT:    or %s11, 0, %s9
  ret float 1.000000e+00
}

; Function Attrs: norecurse nounwind readnone
define double @fcos_double_zero() {
; CHECK-LABEL: fcos_double_zero:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s0, 1072693248
; CHECK-NEXT:    or %s11, 0, %s9
  ret double 1.000000e+00
}

; Function Attrs: nounwind readnone
define fp128 @fcos_quad_zero() {
; CHECK-LABEL: fcos_quad_zero:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s2, .LCPI{{[0-9]+}}_0@hi(, %s0)
; CHECK-NEXT:    ld %s0, 8(, %s2)
; CHECK-NEXT:    ld %s1, (, %s2)
; CHECK-NEXT:    lea %s2, cosl@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s12, cosl@hi(, %s2)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %1 = tail call fast fp128 @llvm.cos.f128(fp128 0xL00000000000000000000000000000000)
  ret fp128 %1
}

; Function Attrs: norecurse nounwind readnone
define float @fcos_float_const() {
; CHECK-LABEL: fcos_float_const:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s0, -1093332685
; CHECK-NEXT:    or %s11, 0, %s9
  ret float 0xBFDAA22660000000
}

; Function Attrs: norecurse nounwind readnone
define double @fcos_double_const() {
; CHECK-LABEL: fcos_double_const:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, 1465086469
; CHECK-NEXT:    lea.sl %s0, -1076190682(, %s0)
; CHECK-NEXT:    or %s11, 0, %s9
  ret double 0xBFDAA22657537205
}

; Function Attrs: nounwind readnone
define fp128 @fcos_quad_const() {
; CHECK-LABEL: fcos_quad_const:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s2, .LCPI{{[0-9]+}}_0@hi(, %s0)
; CHECK-NEXT:    ld %s0, 8(, %s2)
; CHECK-NEXT:    ld %s1, (, %s2)
; CHECK-NEXT:    lea %s2, cosl@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s12, cosl@hi(, %s2)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %1 = tail call fast fp128 @llvm.cos.f128(fp128 0xL0000000000000000C000000000000000)
  ret fp128 %1
}
