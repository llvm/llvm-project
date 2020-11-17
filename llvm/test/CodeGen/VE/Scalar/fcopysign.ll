; RUN: llc < %s -mtriple=ve | FileCheck %s

;;; Test ‘llvm.copysign.*’ Intrinsic
;;;
;;; Syntax:
;;;   This is an overloaded intrinsic. You can use llvm.copysign on any
;;;   floating-point or vector of floating-point type. Not all targets
;;;   support all types however.
;;;
;;; declare float     @llvm.copysign.f32(float  %Mag, float  %Sgn)
;;; declare double    @llvm.copysign.f64(double %Mag, double %Sgn)
;;; declare x86_fp80  @llvm.copysign.f80(x86_fp80  %Mag, x86_fp80  %Sgn)
;;; declare fp128     @llvm.copysign.f128(fp128 %Mag, fp128 %Sgn)
;;; declare ppc_fp128 @llvm.copysign.ppcf128(ppc_fp128  %Mag, ppc_fp128  %Sgn)
;;;
;;; Overview:
;;;   The ‘llvm.copysign.*’ intrinsics return a value with the magnitude of
;;;   the first operand and the sign of the second operand.
;;;
;;; Arguments:
;;;   The arguments and return value are floating-point numbers of the same
;;;   type.
;;;
;;; Semantics:
;;;   This function returns the same values as the libm copysign functions
;;;   would, and handles error conditions in the same way.
;;;
;;; Note:
;;;   We test only float/double/fp128.

; Function Attrs: nounwind readnone
define float @copysign_float_var(float %0, float %1) {
; CHECK-LABEL: copysign_float_var:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sra.l %s1, %s1, 32
; CHECK-NEXT:    lea %s2, -2147483648
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    and %s1, %s1, %s2
; CHECK-NEXT:    sra.l %s0, %s0, 32
; CHECK-NEXT:    and %s0, %s0, (33)0
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    sll %s0, %s0, 32
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast float @llvm.copysign.f32(float %0, float %1)
  ret float %3
}

; Function Attrs: nounwind readnone speculatable willreturn
declare float @llvm.copysign.f32(float, float)

; Function Attrs: nounwind readnone
define double @copysign_double_var(double %0, double %1) {
; CHECK-LABEL: copysign_double_var:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s1, %s1, (1)1
; CHECK-NEXT:    and %s0, %s0, (1)0
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = tail call fast double @llvm.copysign.f64(double %0, double %1)
  ret double %3
}

; Function Attrs: nounwind readnone speculatable willreturn
declare double @llvm.copysign.f64(double, double)

; Function Attrs: nounwind readnone
define fp128 @copysign_quad_var(fp128 %0, fp128 %1) {
; CHECK-LABEL: copysign_quad_var:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    st %s3, 192(, %s11)
; CHECK-NEXT:    st %s2, 200(, %s11)
; CHECK-NEXT:    st %s1, 176(, %s11)
; CHECK-NEXT:    st %s0, 184(, %s11)
; CHECK-NEXT:    ld1b.zx %s0, 207(, %s11)
; CHECK-NEXT:    ld1b.zx %s1, 191(, %s11)
; CHECK-NEXT:    lea %s2, 128
; CHECK-NEXT:    and %s0, %s0, %s2
; CHECK-NEXT:    and %s1, %s1, (57)0
; CHECK-NEXT:    or %s0, %s1, %s0
; CHECK-NEXT:    st1b %s0, 191(, %s11)
; CHECK-NEXT:    ld %s1, 176(, %s11)
; CHECK-NEXT:    ld %s0, 184(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = tail call fast fp128 @llvm.copysign.f128(fp128 %0, fp128 %1)
  ret fp128 %3
}

; Function Attrs: nounwind readnone speculatable willreturn
declare fp128 @llvm.copysign.f128(fp128, fp128)

; Function Attrs: nounwind readnone
define float @copysign_float_zero(float %0) {
; CHECK-LABEL: copysign_float_zero:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sra.l %s0, %s0, 32
; CHECK-NEXT:    lea %s1, -2147483648
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    and %s0, %s0, %s1
; CHECK-NEXT:    sll %s0, %s0, 32
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast float @llvm.copysign.f32(float 0.000000e+00, float %0)
  ret float %2
}

; Function Attrs: nounwind readnone
define double @copysign_double_zero(double %0) {
; CHECK-LABEL: copysign_double_zero:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (1)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast double @llvm.copysign.f64(double 0.000000e+00, double %0)
  ret double %2
}

; Function Attrs: nounwind readnone
define fp128 @copysign_quad_zero(fp128 %0) {
; CHECK-LABEL: copysign_quad_zero:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s2, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s2, .LCPI{{[0-9]+}}_0@hi(, %s2)
; CHECK-NEXT:    ld %s4, 8(, %s2)
; CHECK-NEXT:    ld %s5, (, %s2)
; CHECK-NEXT:    st %s1, 192(, %s11)
; CHECK-NEXT:    st %s0, 200(, %s11)
; CHECK-NEXT:    st %s5, 176(, %s11)
; CHECK-NEXT:    st %s4, 184(, %s11)
; CHECK-NEXT:    ld1b.zx %s0, 207(, %s11)
; CHECK-NEXT:    ld1b.zx %s1, 191(, %s11)
; CHECK-NEXT:    lea %s2, 128
; CHECK-NEXT:    and %s0, %s0, %s2
; CHECK-NEXT:    and %s1, %s1, (57)0
; CHECK-NEXT:    or %s0, %s1, %s0
; CHECK-NEXT:    st1b %s0, 191(, %s11)
; CHECK-NEXT:    ld %s1, 176(, %s11)
; CHECK-NEXT:    ld %s0, 184(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = tail call fast fp128 @llvm.copysign.f128(fp128 0xL00000000000000000000000000000000, fp128 %0)
  ret fp128 %2
}

; Function Attrs: nounwind readnone
define float @copysign_float_const(float %0) {
; CHECK-LABEL: copysign_float_const:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sra.l %s0, %s0, 32
; CHECK-NEXT:    lea %s1, -2147483648
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    and %s0, %s0, %s1
; CHECK-NEXT:    lea %s1, 1073741824
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    sll %s0, %s0, 32
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast float @llvm.copysign.f32(float -2.000000e+00, float %0)
  ret float %2
}

; Function Attrs: nounwind readnone
define double @copysign_double_const(double %0) {
; CHECK-LABEL: copysign_double_const:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (1)1
; CHECK-NEXT:    lea.sl %s1, 1073741824
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = tail call fast double @llvm.copysign.f64(double -2.000000e+00, double %0)
  ret double %2
}

; Function Attrs: nounwind readnone
define fp128 @copysign_quad_const(fp128 %0) {
; CHECK-LABEL: copysign_quad_const:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s2, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s2, .LCPI{{[0-9]+}}_0@hi(, %s2)
; CHECK-NEXT:    ld %s4, 8(, %s2)
; CHECK-NEXT:    ld %s5, (, %s2)
; CHECK-NEXT:    st %s1, 192(, %s11)
; CHECK-NEXT:    st %s0, 200(, %s11)
; CHECK-NEXT:    st %s5, 176(, %s11)
; CHECK-NEXT:    st %s4, 184(, %s11)
; CHECK-NEXT:    ld1b.zx %s0, 207(, %s11)
; CHECK-NEXT:    ld1b.zx %s1, 191(, %s11)
; CHECK-NEXT:    lea %s2, 128
; CHECK-NEXT:    and %s0, %s0, %s2
; CHECK-NEXT:    and %s1, %s1, (57)0
; CHECK-NEXT:    or %s0, %s1, %s0
; CHECK-NEXT:    st1b %s0, 191(, %s11)
; CHECK-NEXT:    ld %s1, 176(, %s11)
; CHECK-NEXT:    ld %s0, 184(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = tail call fast fp128 @llvm.copysign.f128(fp128 0xL0000000000000000C000000000000000, fp128 %0)
  ret fp128 %2
}
