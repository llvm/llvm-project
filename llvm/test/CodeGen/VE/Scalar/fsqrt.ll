; RUN: llc < %s -mtriple=ve | FileCheck %s

;;; Test ‘llvm.sqrt.*’ intrinsic
;;;
;;; Syntax:
;;;   This is an overloaded intrinsic. You can use llvm.sqrt on any
;;;   floating-point or vector of floating-point type. Not all targets
;;;   support all types however.
;;;
;;; declare float     @llvm.sqrt.f32(float %Val)
;;; declare double    @llvm.sqrt.f64(double %Val)
;;; declare x86_fp80  @llvm.sqrt.f80(x86_fp80 %Val)
;;; declare fp128     @llvm.sqrt.f128(fp128 %Val)
;;; declare ppc_fp128 @llvm.sqrt.ppcf128(ppc_fp128 %Val)
;;;
;;; Overview:
;;;   The ‘llvm.sqrt’ intrinsics return the square root of the specified value.
;;;
;;; Arguments:
;;;   The argument and return value are floating-point numbers of the same type.
;;;
;;; Semantics:
;;;   Return the same value as a corresponding libm ‘sqrt’ function but without
;;;   trapping or setting errno. For types specified by IEEE-754, the result
;;;   matches a conforming libm implementation.
;;;
;;;   When specified with the fast-math-flag ‘afn’, the result may be
;;;   approximated using a less accurate calculation.
;;;
;;; Note:
;;;   We test only float/double/fp128.

; Function Attrs: nounwind readnone
define float @fsqrt_float_var(float %0) {
; CHECK-LABEL: fsqrt_float_var:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s1, sqrtf@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s12, sqrtf@hi(, %s1)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = tail call fast float @llvm.sqrt.f32(float %0)
  ret float %2
}

; Function Attrs: nounwind readnone speculatable willreturn
declare float @llvm.sqrt.f32(float)

; Function Attrs: nounwind readnone
define double @fsqrt_double_var(double %0) {
; CHECK-LABEL: fsqrt_double_var:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s1, sqrt@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s12, sqrt@hi(, %s1)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = tail call fast double @llvm.sqrt.f64(double %0)
  ret double %2
}

; Function Attrs: nounwind readnone speculatable willreturn
declare double @llvm.sqrt.f64(double)

; Function Attrs: nounwind readnone
define fp128 @fsqrt_quad_var(fp128 %0) {
; CHECK-LABEL: fsqrt_quad_var:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s2, sqrtl@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s12, sqrtl@hi(, %s2)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = tail call fast fp128 @llvm.sqrt.f128(fp128 %0)
  ret fp128 %2
}

; Function Attrs: nounwind readnone speculatable willreturn
declare fp128 @llvm.sqrt.f128(fp128)

; Function Attrs: norecurse nounwind readnone
define float @fsqrt_float_zero() {
; CHECK-LABEL: fsqrt_float_zero:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea.sl %s0, 0
; CHECK-NEXT:    b.l.t (, %s10)
  ret float 0.000000e+00
}

; Function Attrs: norecurse nounwind readnone
define double @fsqrt_double_zero() {
; CHECK-LABEL: fsqrt_double_zero:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea.sl %s0, 0
; CHECK-NEXT:    b.l.t (, %s10)
  ret double 0.000000e+00
}

; Function Attrs: nounwind readnone
define fp128 @fsqrt_quad_zero() {
; CHECK-LABEL: fsqrt_quad_zero:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s2, .LCPI{{[0-9]+}}_0@hi(, %s0)
; CHECK-NEXT:    ld %s0, 8(, %s2)
; CHECK-NEXT:    ld %s1, (, %s2)
; CHECK-NEXT:    lea %s2, sqrtl@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s12, sqrtl@hi(, %s2)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %1 = tail call fast fp128 @llvm.sqrt.f128(fp128 0xL00000000000000000000000000000000)
  ret fp128 %1
}

; Function Attrs: norecurse nounwind readnone
define float @fsqrt_float_const() {
; CHECK-LABEL: fsqrt_float_const:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea.sl %s0, -4194304
; CHECK-NEXT:    b.l.t (, %s10)
  ret float 0xFFF8000000000000
}

; Function Attrs: norecurse nounwind readnone
define double @fsqrt_double_const() {
; CHECK-LABEL: fsqrt_double_const:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea.sl %s0, -524288
; CHECK-NEXT:    b.l.t (, %s10)
  ret double 0xFFF8000000000000
}

; Function Attrs: nounwind readnone
define fp128 @fsqrt_quad_const() {
; CHECK-LABEL: fsqrt_quad_const:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s2, .LCPI{{[0-9]+}}_0@hi(, %s0)
; CHECK-NEXT:    ld %s0, 8(, %s2)
; CHECK-NEXT:    ld %s1, (, %s2)
; CHECK-NEXT:    lea %s2, sqrtl@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s12, sqrtl@hi(, %s2)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %1 = tail call fast fp128 @llvm.sqrt.f128(fp128 0xL0000000000000000C000000000000000)
  ret fp128 %1
}
