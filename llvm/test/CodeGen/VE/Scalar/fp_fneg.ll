; RUN: llc < %s -mtriple=ve | FileCheck %s

;;; Test ‘fneg’ Instruction
;;;
;;; Syntax:
;;;   <result> = fneg [fast-math flags]* <ty> <op1>   ; yields ty:result
;;;
;;; Overview:
;;;    The ‘fneg’ instruction returns the negation of its operand.
;;;
;;; Arguments:
;;;   The argument to the ‘fneg’ instruction must be a floating-point or
;;;   vector of floating-point values.
;;;
;;; Semantics:
;;;
;;;   The value produced is a copy of the operand with its sign bit flipped.
;;;   This instruction can also take any number of fast-math flags, which are
;;;   optimization hints to enable otherwise unsafe floating-point
;;;   optimizations.
;;;
;;; Example:
;;;   <result> = fneg float %val          ; yields float:result = -%var
;;;
;;; Note:
;;;   We test only float/double/fp128.

; Function Attrs: norecurse nounwind readnone
define float @fneg_float(float %0) {
; CHECK-LABEL: fneg_float:
; CHECK:       # %bb.0:
; CHECK-NEXT:    sra.l %s0, %s0, 32
; CHECK-NEXT:    lea %s1, -2147483648
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    xor %s0, %s0, %s1
; CHECK-NEXT:    sll %s0, %s0, 32
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = fneg float %0
  ret float %2
}

; Function Attrs: norecurse nounwind readnone
define double @fneg_double(double %0) {
; CHECK-LABEL: fneg_double:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xor %s0, %s0, (1)1
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = fneg double %0
  ret double %2
}

; Function Attrs: norecurse nounwind readnone
define fp128 @fneg_quad(fp128 %0) {
; CHECK-LABEL: fneg_quad:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    st %s1, 176(, %s11)
; CHECK-NEXT:    st %s0, 184(, %s11)
; CHECK-NEXT:    ld1b.zx %s0, 191(, %s11)
; CHECK-NEXT:    lea %s1, 128
; CHECK-NEXT:    xor %s0, %s0, %s1
; CHECK-NEXT:    st1b %s0, 191(, %s11)
; CHECK-NEXT:    ld %s1, 176(, %s11)
; CHECK-NEXT:    ld %s0, 184(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = fneg fp128 %0
  ret fp128 %2
}
