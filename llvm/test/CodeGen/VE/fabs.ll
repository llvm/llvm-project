; RUN: llc < %s -mtriple=ve | FileCheck %s

;;; Test ‘llvm.fabs.*’ Intrinsic
;;;
;;; Syntax:
;;;   This is an overloaded intrinsic. You can use llvm.fabs on any
;;;   floating-point or vector of floating-point type. Not all targets
;;;   support all types however.
;;;
;;; declare float     @llvm.fabs.f32(float  %Val)
;;; declare double    @llvm.fabs.f64(double %Val)
;;; declare x86_fp80  @llvm.fabs.f80(x86_fp80 %Val)
;;; declare fp128     @llvm.fabs.f128(fp128 %Val)
;;; declare ppc_fp128 @llvm.fabs.ppcf128(ppc_fp128 %Val)
;;;
;;; Overview:
;;;   The ‘llvm.fabs.*’ intrinsics return the absolute value of the operand.
;;;
;;; Arguments:
;;;   The argument and return value are floating-point numbers of the same
;;;   type.
;;;
;;; Semantics:
;;;   This function returns the same values as the libm fabs functions would,
;;;   and handles error conditions in the same way.
;;;
;;; Note:
;;;   We test only float/double/fp128.

; Function Attrs: nounwind readnone
define float @fabs_float_var(float %0) {
; CHECK-LABEL: fabs_float_var:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sra.l %s0, %s0, 32
; CHECK-NEXT:    and %s0, %s0, (33)0
; CHECK-NEXT:    sll %s0, %s0, 32
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = tail call fast float @llvm.fabs.f32(float %0)
  ret float %2
}

; Function Attrs: nounwind readnone speculatable willreturn
declare float @llvm.fabs.f32(float)

; Function Attrs: nounwind readnone
define double @fabs_double_var(double %0) {
; CHECK-LABEL: fabs_double_var:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s0, %s0, (1)0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = tail call fast double @llvm.fabs.f64(double %0)
  ret double %2
}

; Function Attrs: nounwind readnone speculatable willreturn
declare double @llvm.fabs.f64(double)

; Function Attrs: nounwind readnone
define fp128 @fabs_quad_var(fp128 %0) {
; CHECK-LABEL: fabs_quad_var:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    st %s1, 176(, %s11)
; CHECK-NEXT:    st %s0, 184(, %s11)
; CHECK-NEXT:    ld1b.zx %s0, 191(, %s11)
; CHECK-NEXT:    and %s0, %s0, (57)0
; CHECK-NEXT:    st1b %s0, 191(, %s11)
; CHECK-NEXT:    ld %s1, 176(, %s11)
; CHECK-NEXT:    ld %s0, 184(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = tail call fast fp128 @llvm.fabs.f128(fp128 %0)
  ret fp128 %2
}

; Function Attrs: nounwind readnone speculatable willreturn
declare fp128 @llvm.fabs.f128(fp128)

; Function Attrs: norecurse nounwind readnone
define float @fabs_float_zero() {
; CHECK-LABEL: fabs_float_zero:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s0, 0
; CHECK-NEXT:    or %s11, 0, %s9
  ret float 0.000000e+00
}

; Function Attrs: norecurse nounwind readnone
define double @fabs_double_zero() {
; CHECK-LABEL: fabs_double_zero:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s0, 0
; CHECK-NEXT:    or %s11, 0, %s9
  ret double 0.000000e+00
}

; Function Attrs: norecurse nounwind readnone
define fp128 @fabs_quad_zero() {
; CHECK-LABEL: fabs_quad_zero:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s2, .LCPI{{[0-9]+}}_0@hi(, %s0)
; CHECK-NEXT:    ld %s0, 8(, %s2)
; CHECK-NEXT:    ld %s1, (, %s2)
; CHECK-NEXT:    or %s11, 0, %s9
  ret fp128 0xL00000000000000000000000000000000
}

; Function Attrs: norecurse nounwind readnone
define float @fabs_float_const() {
; CHECK-LABEL: fabs_float_const:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s0, 1073741824
; CHECK-NEXT:    or %s11, 0, %s9
  ret float 2.000000e+00
}

; Function Attrs: norecurse nounwind readnone
define double @fabs_double_const() {
; CHECK-LABEL: fabs_double_const:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s0, 1073741824
; CHECK-NEXT:    or %s11, 0, %s9
  ret double 2.000000e+00
}

; Function Attrs: nounwind readnone
define fp128 @fabs_quad_const() {
; CHECK-LABEL: fabs_quad_const:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s2, .LCPI{{[0-9]+}}_0@hi(, %s0)
; CHECK-NEXT:    ld %s0, 8(, %s2)
; CHECK-NEXT:    ld %s1, (, %s2)
; CHECK-NEXT:    or %s11, 0, %s9
  %1 = tail call fast fp128 @llvm.fabs.f128(fp128 0xL0000000000000000C000000000000000)
  ret fp128 %1
}
