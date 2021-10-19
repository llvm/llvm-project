; RUN: llc < %s -mtriple=ve | FileCheck %s

;;; Test ‘llvm.sin.*’ intrinsic
;;;
;;; Syntax:
;;;   This is an overloaded intrinsic. You can use llvm.sin on any
;;;   floating-point or vector of floating-point type. Not all targets
;;;   support all types however.
;;;
;;; declare float     @llvm.sin.f32(float  %Val)
;;; declare double    @llvm.sin.f64(double %Val)
;;; declare x86_fp80  @llvm.sin.f80(x86_fp80  %Val)
;;; declare fp128     @llvm.sin.f128(fp128 %Val)
;;; declare ppc_fp128 @llvm.sin.ppcf128(ppc_fp128  %Val)
;;;
;;; Overview:
;;;   The ‘llvm.sin.*’ intrinsics return the sine of the operand.
;;;
;;; Arguments:
;;;   The argument and return value are floating-point numbers of the same type.
;;;
;;; Semantics:
;;;   Return the same value as a corresponding libm ‘sin’ function but without
;;;   trapping or setting errno.
;;;
;;;   When specified with the fast-math-flag ‘afn’, the result may be
;;;   approximated using a less accurate calculation.
;;;
;;; Note:
;;;   We test only float/double/fp128.

; Function Attrs: nounwind readnone
define float @fsin_float_var(float %0) {
; CHECK-LABEL: fsin_float_var:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s1, sinf@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s12, sinf@hi(, %s1)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = tail call fast float @llvm.sin.f32(float %0)
  ret float %2
}

; Function Attrs: nounwind readnone speculatable willreturn
declare float @llvm.sin.f32(float)

; Function Attrs: nounwind readnone
define double @fsin_double_var(double %0) {
; CHECK-LABEL: fsin_double_var:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s1, sin@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s12, sin@hi(, %s1)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = tail call fast double @llvm.sin.f64(double %0)
  ret double %2
}

; Function Attrs: nounwind readnone speculatable willreturn
declare double @llvm.sin.f64(double)

; Function Attrs: nounwind readnone
define fp128 @fsin_quad_var(fp128 %0) {
; CHECK-LABEL: fsin_quad_var:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s2, sinl@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s12, sinl@hi(, %s2)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = tail call fast fp128 @llvm.sin.f128(fp128 %0)
  ret fp128 %2
}

; Function Attrs: nounwind readnone speculatable willreturn
declare fp128 @llvm.sin.f128(fp128)

; Function Attrs: norecurse nounwind readnone
define float @fsin_float_zero() {
; CHECK-LABEL: fsin_float_zero:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea.sl %s0, 0
; CHECK-NEXT:    b.l.t (, %s10)
  ret float 0.000000e+00
}

; Function Attrs: norecurse nounwind readnone
define double @fsin_double_zero() {
; CHECK-LABEL: fsin_double_zero:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea.sl %s0, 0
; CHECK-NEXT:    b.l.t (, %s10)
  ret double 0.000000e+00
}

; Function Attrs: nounwind readnone
define fp128 @fsin_quad_zero() {
; CHECK-LABEL: fsin_quad_zero:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s2, .LCPI{{[0-9]+}}_0@hi(, %s0)
; CHECK-NEXT:    ld %s0, 8(, %s2)
; CHECK-NEXT:    ld %s1, (, %s2)
; CHECK-NEXT:    lea %s2, sinl@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s12, sinl@hi(, %s2)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %1 = tail call fast fp128 @llvm.sin.f128(fp128 0xL00000000000000000000000000000000)
  ret fp128 %1
}

; Function Attrs: norecurse nounwind readnone
define float @fsin_float_const() {
; CHECK-LABEL: fsin_float_const:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea.sl %s0, -1083652169
; CHECK-NEXT:    b.l.t (, %s10)
  ret float 0xBFED18F6E0000000
}

; Function Attrs: norecurse nounwind readnone
define double @fsin_double_const() {
; CHECK-LABEL: fsin_double_const:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, -355355578
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, -1074980618(, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  ret double 0xBFED18F6EAD1B446
}

; Function Attrs: nounwind readnone
define fp128 @fsin_quad_const() {
; CHECK-LABEL: fsin_quad_const:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s2, .LCPI{{[0-9]+}}_0@hi(, %s0)
; CHECK-NEXT:    ld %s0, 8(, %s2)
; CHECK-NEXT:    ld %s1, (, %s2)
; CHECK-NEXT:    lea %s2, sinl@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s12, sinl@hi(, %s2)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %1 = tail call fast fp128 @llvm.sin.f128(fp128 0xL0000000000000000C000000000000000)
  ret fp128 %1
}
