; RUN: llc < %s -mtriple=ve | FileCheck %s

;;; Test ‘frem’ Instruction
;;;
;;; Syntax:
;;;   <result> = frem ptr <ty> <op1>, <op2> ; yields ty:result
;;;
;;; Overview:
;;;   The ‘frem’ instruction returns the remainder from the division of its two
;;;   operands.
;;;
;;; Arguments:
;;;   The two arguments to the ‘frem’ instruction must be floating-point or
;;;   vector of floating-point values. Both arguments must have identical types.
;;;
;;; Semantics:
;;;   The value produced is the floating-point remainder of the two operands.
;;;   This is the same output as a libm ‘fmod’ function, but without any
;;;   possibility of setting errno. The remainder has the same sign as the
;;;   dividend. This instruction is assumed to execute in the default
;;;   floating-point environment. This instruction can also take any number
;;;   of fast-math flags, which are optimization hints to enable otherwise
;;;   unsafe floating-point optimizations:
;;;
;;; Example:
;;;
;;;   <result> = frem float 4.0, %var ; yields float:result = 4.0 % %var
;;;
;;; Note:
;;;   We test only float/double/fp128.
;;;   We have no way to generated frem from C source code, so convert fdiv
;;;   to frem by using sed program.

; Function Attrs: norecurse nounwind readnone
define float @frem_float_var(float %0, float %1) {
; CHECK-LABEL: frem_float_var:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s2, fmodf@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s12, fmodf@hi(, %s2)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = frem float %0, %1
  ret float %3
}

; Function Attrs: norecurse nounwind readnone
define double @frem_double_var(double %0, double %1) {
; CHECK-LABEL: frem_double_var:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s2, fmod@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s12, fmod@hi(, %s2)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = frem double %0, %1
  ret double %3
}

; Function Attrs: norecurse nounwind readnone
define fp128 @frem_quad_var(fp128 %0, fp128 %1) {
; CHECK-LABEL: frem_quad_var:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s4, fmodl@lo
; CHECK-NEXT:    and %s4, %s4, (32)0
; CHECK-NEXT:    lea.sl %s12, fmodl@hi(, %s4)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = frem fp128 %0, %1
  ret fp128 %3
}

; Function Attrs: norecurse nounwind readnone
define float @frem_float_zero(float %0) {
; CHECK-LABEL: frem_float_zero:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 0, %s0
; CHECK-NEXT:    lea %s0, fmodf@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, fmodf@hi(, %s0)
; CHECK-NEXT:    lea.sl %s0, 0
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = frem float 0.000000e+00, %0
  ret float %2
}

; Function Attrs: norecurse nounwind readnone
define double @frem_double_zero(double %0) {
; CHECK-LABEL: frem_double_zero:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 0, %s0
; CHECK-NEXT:    lea %s0, fmod@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, fmod@hi(, %s0)
; CHECK-NEXT:    lea.sl %s0, 0
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = frem double 0.000000e+00, %0
  ret double %2
}

; Function Attrs: norecurse nounwind readnone
define fp128 @frem_quad_zero(fp128 %0) {
; CHECK-LABEL: frem_quad_zero:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s2, 0, %s0
; CHECK-NEXT:    or %s3, 0, %s1
; CHECK-NEXT:    lea %s0, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s4, .LCPI{{[0-9]+}}_0@hi(, %s0)
; CHECK-NEXT:    ld %s0, 8(, %s4)
; CHECK-NEXT:    ld %s1, (, %s4)
; CHECK-NEXT:    lea %s4, fmodl@lo
; CHECK-NEXT:    and %s4, %s4, (32)0
; CHECK-NEXT:    lea.sl %s12, fmodl@hi(, %s4)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = frem fp128 0xL00000000000000000000000000000000, %0
  ret fp128 %2
}

; Function Attrs: norecurse nounwind readnone
define float @frem_float_cont(float %0) {
; CHECK-LABEL: frem_float_cont:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 0, %s0
; CHECK-NEXT:    lea %s0, fmodf@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, fmodf@hi(, %s0)
; CHECK-NEXT:    lea.sl %s0, -1073741824
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = frem float -2.000000e+00, %0
  ret float %2
}

; Function Attrs: norecurse nounwind readnone
define double @frem_double_cont(double %0) {
; CHECK-LABEL: frem_double_cont:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 0, %s0
; CHECK-NEXT:    lea %s0, fmod@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, fmod@hi(, %s0)
; CHECK-NEXT:    lea.sl %s0, -1073741824
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = frem double -2.000000e+00, %0
  ret double %2
}

; Function Attrs: norecurse nounwind readnone
define fp128 @frem_quad_cont(fp128 %0) {
; CHECK-LABEL: frem_quad_cont:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s2, 0, %s0
; CHECK-NEXT:    or %s3, 0, %s1
; CHECK-NEXT:    lea %s0, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s4, .LCPI{{[0-9]+}}_0@hi(, %s0)
; CHECK-NEXT:    ld %s0, 8(, %s4)
; CHECK-NEXT:    ld %s1, (, %s4)
; CHECK-NEXT:    lea %s4, fmodl@lo
; CHECK-NEXT:    and %s4, %s4, (32)0
; CHECK-NEXT:    lea.sl %s12, fmodl@hi(, %s4)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = frem fp128 0xL0000000000000000C000000000000000, %0
  ret fp128 %2
}
