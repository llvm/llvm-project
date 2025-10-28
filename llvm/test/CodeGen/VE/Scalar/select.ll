; RUN: llc < %s -mtriple=ve | FileCheck %s

;;; Test ‘select’ instruction
;;;
;;; Syntax:
;;;   <result> = select [fast-math flags] selty <cond>, <ty> <val1>, <ty> <val2>
;;;                                             ; yields ty
;;;
;;;   selty is either i1 or {<N x i1>}
;;;
;;; Overview:
;;;   The ‘select’ instruction is used to choose one value based on a condition,
;;;   without IR-level branching.
;;;
;;; Arguments:
;;;   The ‘select’ instruction requires an ‘i1’ value or a vector of ‘i1’ values
;;;   indicating the condition, and two values of the same first class type.
;;;
;;;   The optional fast-math flags marker indicates that the select has one or
;;;   more fast-math flags. These are optimization hints to enable otherwise
;;;   unsafe floating-point optimizations. Fast-math flags are only valid for
;;;   selects that return a floating-point scalar or vector type, or an array
;;;   (nested to any depth) of floating-point scalar or vector types.
;;;
;;; Semantics:
;;;   If the condition is an i1 and it evaluates to 1, the instruction returns
;;;   the first value argument; otherwise, it returns the second value argument.
;;;
;;;   If the condition is a vector of i1, then the value arguments must be
;;;   vectors of the same size, and the selection is done element by element.
;;;
;;;   If the condition is an i1 and the value arguments are vectors of the same
;;;   size, then an entire vector is selected.
;;;
;;; Example:
;;;   %X = select i1 true, i8 17, i8 42 ; yields i8:17
;;;
;;; Note:
;;;   We test only i1/i8/u8/i16/u16/i32/u32/i64/u64/i128/u128/float/double/fp128

; Function Attrs: norecurse nounwind readnone
define zeroext i1 @select_i1_var(i1 zeroext %0, i1 zeroext %1, i1 zeroext %2) {
; CHECK-LABEL: select_i1_var:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s2, %s1, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s2, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = select i1 %0, i1 %1, i1 %2
  ret i1 %4
}

; Function Attrs: norecurse nounwind readnone
define signext i8 @select_i8_var(i1 zeroext %0, i8 signext %1, i8 signext %2) {
; CHECK-LABEL: select_i8_var:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s2, %s1, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s2, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = select i1 %0, i8 %1, i8 %2
  ret i8 %4
}

; Function Attrs: norecurse nounwind readnone
define zeroext i8 @select_u8_var(i1 zeroext %0, i8 zeroext %1, i8 zeroext %2) {
; CHECK-LABEL: select_u8_var:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s2, %s1, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s2, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = select i1 %0, i8 %1, i8 %2
  ret i8 %4
}

; Function Attrs: norecurse nounwind readnone
define signext i16 @select_i16_var(i1 zeroext %0, i16 signext %1, i16 signext %2) {
; CHECK-LABEL: select_i16_var:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s2, %s1, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s2, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = select i1 %0, i16 %1, i16 %2
  ret i16 %4
}

; Function Attrs: norecurse nounwind readnone
define zeroext i16 @select_u16_var(i1 zeroext %0, i16 zeroext %1, i16 zeroext %2) {
; CHECK-LABEL: select_u16_var:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s2, %s1, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s2, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = select i1 %0, i16 %1, i16 %2
  ret i16 %4
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @select_i32_var(i1 zeroext %0, i32 signext %1, i32 signext %2) {
; CHECK-LABEL: select_i32_var:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s2, %s1, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s2, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = select i1 %0, i32 %1, i32 %2
  ret i32 %4
}

; Function Attrs: norecurse nounwind readnone
define zeroext i32 @select_u32_var(i1 zeroext %0, i32 zeroext %1, i32 zeroext %2) {
; CHECK-LABEL: select_u32_var:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s2, %s1, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s2, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = select i1 %0, i32 %1, i32 %2
  ret i32 %4
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_i64_var(i1 zeroext %0, i64 %1, i64 %2) {
; CHECK-LABEL: select_i64_var:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s2, %s1, %s0
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = select i1 %0, i64 %1, i64 %2
  ret i64 %4
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_u64_var(i1 zeroext %0, i64 %1, i64 %2) {
; CHECK-LABEL: select_u64_var:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s2, %s1, %s0
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = select i1 %0, i64 %1, i64 %2
  ret i64 %4
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_i128_var(i1 zeroext %0, i128 %1, i128 %2) {
; CHECK-LABEL: select_i128_var:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s3, %s1, %s0
; CHECK-NEXT:    cmov.w.ne %s4, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s1, 0, %s4
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = select i1 %0, i128 %1, i128 %2
  ret i128 %4
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_u128_var(i1 zeroext %0, i128 %1, i128 %2) {
; CHECK-LABEL: select_u128_var:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s3, %s1, %s0
; CHECK-NEXT:    cmov.w.ne %s4, %s2, %s0
; CHECK-NEXT:    or %s0, 0, %s3
; CHECK-NEXT:    or %s1, 0, %s4
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = select i1 %0, i128 %1, i128 %2
  ret i128 %4
}

; Function Attrs: norecurse nounwind readnone
define float @select_float_var(i1 zeroext %0, float %1, float %2) {
; CHECK-LABEL: select_float_var:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s2, %s1, %s0
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = select fast i1 %0, float %1, float %2
  ret float %4
}

; Function Attrs: norecurse nounwind readnone
define double @select_double_var(i1 zeroext %0, double %1, double %2) {
; CHECK-LABEL: select_double_var:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s2, %s1, %s0
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = select fast i1 %0, double %1, double %2
  ret double %4
}

; Function Attrs: norecurse nounwind readnone
define fp128 @select_quad_var(i1 zeroext %0, fp128 %1, fp128 %2) {
; CHECK-LABEL: select_quad_var:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s4, %s2, %s0
; CHECK-NEXT:    cmov.w.ne %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    b.l.t (, %s10)
  %4 = select fast i1 %0, fp128 %1, fp128 %2
  ret fp128 %4
}

; Function Attrs: norecurse nounwind readnone
define zeroext i1 @select_i1_mimm(i1 zeroext %0, i1 zeroext %1) {
; CHECK-LABEL: select_i1_mimm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = or i1 %0, %1
  ret i1 %3
}

; Function Attrs: norecurse nounwind readnone
define signext i8 @select_i8_mimm(i1 zeroext %0, i8 signext %1) {
; CHECK-LABEL: select_i8_mimm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s1, (57)1, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = select i1 %0, i8 -128, i8 %1
  ret i8 %3
}

; Function Attrs: norecurse nounwind readnone
define zeroext i8 @select_u8_mimm(i1 zeroext %0, i8 zeroext %1) {
; CHECK-LABEL: select_u8_mimm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s1, (57)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = select i1 %0, i8 127, i8 %1
  ret i8 %3
}

; Function Attrs: norecurse nounwind readnone
define signext i16 @select_i16_mimm(i1 zeroext %0, i16 signext %1) {
; CHECK-LABEL: select_i16_mimm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s1, (49)1, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = select i1 %0, i16 -32768, i16 %1
  ret i16 %3
}

; Function Attrs: norecurse nounwind readnone
define zeroext i16 @select_u16_mimm(i1 zeroext %0, i16 zeroext %1) {
; CHECK-LABEL: select_u16_mimm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s1, (49)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = select i1 %0, i16 32767, i16 %1
  ret i16 %3
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @select_i32_mimm(i1 zeroext %0, i32 signext %1) {
; CHECK-LABEL: select_i32_mimm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s1, (48)0, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = select i1 %0, i32 65535, i32 %1
  ret i32 %3
}

; Function Attrs: norecurse nounwind readnone
define zeroext i32 @select_u32_mimm(i1 zeroext %0, i32 zeroext %1) {
; CHECK-LABEL: select_u32_mimm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s1, (48)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = select i1 %0, i32 65535, i32 %1
  ret i32 %3
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_i64_mimm(i1 zeroext %0, i64 %1) {
; CHECK-LABEL: select_i64_mimm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s1, (48)0, %s0
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = select i1 %0, i64 65535, i64 %1
  ret i64 %3
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_u64_mimm(i1 zeroext %0, i64 %1) {
; CHECK-LABEL: select_u64_mimm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s1, (48)0, %s0
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = select i1 %0, i64 65535, i64 %1
  ret i64 %3
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_i128_mimm(i1 zeroext %0, i128 %1) {
; CHECK-LABEL: select_i128_mimm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s1, (48)0, %s0
; CHECK-NEXT:    cmov.w.ne %s2, (0)1, %s0
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    or %s1, 0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = select i1 %0, i128 65535, i128 %1
  ret i128 %3
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_u128_mimm(i1 zeroext %0, i128 %1) {
; CHECK-LABEL: select_u128_mimm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s1, (48)0, %s0
; CHECK-NEXT:    cmov.w.ne %s2, (0)1, %s0
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    or %s1, 0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = select i1 %0, i128 65535, i128 %1
  ret i128 %3
}

; Function Attrs: norecurse nounwind readnone
define float @select_float_mimm(i1 zeroext %0, float %1) {
; CHECK-LABEL: select_float_mimm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s1, (2)1, %s0
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = select i1 %0, float -2.000000e+00, float %1
  ret float %3
}

; Function Attrs: norecurse nounwind readnone
define double @select_double_mimm(i1 zeroext %0, double %1) {
; CHECK-LABEL: select_double_mimm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.ne %s1, (2)1, %s0
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = select fast i1 %0, double -2.000000e+00, double %1
  ret double %3
}

; Function Attrs: norecurse nounwind readnone
define fp128 @select_quad_mimm(i1 zeroext %0, fp128 %1) {
; CHECK-LABEL: select_quad_mimm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s1, .LCPI{{[0-9]+}}_0@hi(, %s1)
; CHECK-NEXT:    ld %s4, 8(, %s1)
; CHECK-NEXT:    ld %s1, (, %s1)
; CHECK-NEXT:    cmov.w.ne %s2, %s4, %s0
; CHECK-NEXT:    cmov.w.ne %s3, %s1, %s0
; CHECK-NEXT:    or %s0, 0, %s2
; CHECK-NEXT:    or %s1, 0, %s3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = select fast i1 %0, fp128 0xL0000000000000000C000000000000000, fp128 %1
  ret fp128 %3
}

; Function Attrs: norecurse nounwind readnone
define zeroext i1 @select_mimm_i1(i1 zeroext %0, i1 zeroext %1) {
; CHECK-LABEL: select_mimm_i1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xor %s0, 1, %s0
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = xor i1 %0, true
  %4 = or i1 %3, %1
  ret i1 %4
}

; Function Attrs: norecurse nounwind readnone
define signext i8 @select_mimm_i8(i1 zeroext %0, i8 signext %1) {
; CHECK-LABEL: select_mimm_i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.eq %s1, (57)1, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = select i1 %0, i8 %1, i8 -128
  ret i8 %3
}

; Function Attrs: norecurse nounwind readnone
define zeroext i8 @select_mimm_u8(i1 zeroext %0, i8 zeroext %1) {
; CHECK-LABEL: select_mimm_u8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.eq %s1, (57)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = select i1 %0, i8 %1, i8 127
  ret i8 %3
}

; Function Attrs: norecurse nounwind readnone
define signext i16 @select_mimm_i16(i1 zeroext %0, i16 signext %1) {
; CHECK-LABEL: select_mimm_i16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.eq %s1, (49)1, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = select i1 %0, i16 %1, i16 -32768
  ret i16 %3
}

; Function Attrs: norecurse nounwind readnone
define zeroext i16 @select_mimm_u16(i1 zeroext %0, i16 zeroext %1) {
; CHECK-LABEL: select_mimm_u16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.eq %s1, (49)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = select i1 %0, i16 %1, i16 32767
  ret i16 %3
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @select_mimm_i32(i1 zeroext %0, i32 signext %1) {
; CHECK-LABEL: select_mimm_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.eq %s1, (48)0, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = select i1 %0, i32 %1, i32 65535
  ret i32 %3
}

; Function Attrs: norecurse nounwind readnone
define zeroext i32 @select_mimm_u32(i1 zeroext %0, i32 zeroext %1) {
; CHECK-LABEL: select_mimm_u32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.eq %s1, (48)0, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s1, (0)1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = select i1 %0, i32 %1, i32 65535
  ret i32 %3
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_mimm_i64(i1 zeroext %0, i64 %1) {
; CHECK-LABEL: select_mimm_i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.eq %s1, (48)0, %s0
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = select i1 %0, i64 %1, i64 65535
  ret i64 %3
}

; Function Attrs: norecurse nounwind readnone
define i64 @select_mimm_u64(i1 zeroext %0, i64 %1) {
; CHECK-LABEL: select_mimm_u64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.eq %s1, (48)0, %s0
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = select i1 %0, i64 %1, i64 65535
  ret i64 %3
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_mimm_i128(i1 zeroext %0, i128 %1) {
; CHECK-LABEL: select_mimm_i128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.eq %s1, (48)0, %s0
; CHECK-NEXT:    cmov.w.eq %s2, (0)1, %s0
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    or %s1, 0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = select i1 %0, i128 %1, i128 65535
  ret i128 %3
}

; Function Attrs: norecurse nounwind readnone
define i128 @select_mimm_u128(i1 zeroext %0, i128 %1) {
; CHECK-LABEL: select_mimm_u128:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.eq %s1, (48)0, %s0
; CHECK-NEXT:    cmov.w.eq %s2, (0)1, %s0
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    or %s1, 0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = select i1 %0, i128 %1, i128 65535
  ret i128 %3
}

; Function Attrs: norecurse nounwind readnone
define float @select_mimm_float(i1 zeroext %0, float %1) {
; CHECK-LABEL: select_mimm_float:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.eq %s1, (2)1, %s0
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = select i1 %0, float %1, float -2.000000e+00
  ret float %3
}

; Function Attrs: norecurse nounwind readnone
define double @select_mimm_double(i1 zeroext %0, double %1) {
; CHECK-LABEL: select_mimm_double:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cmov.w.eq %s1, (2)1, %s0
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = select fast i1 %0, double %1, double -2.000000e+00
  ret double %3
}

; Function Attrs: norecurse nounwind readnone
define fp128 @select_mimm_quad(i1 zeroext %0, fp128 %1) {
; CHECK-LABEL: select_mimm_quad:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s1, .LCPI{{[0-9]+}}_0@hi(, %s1)
; CHECK-NEXT:    ld %s4, 8(, %s1)
; CHECK-NEXT:    ld %s5, (, %s1)
; CHECK-NEXT:    cmov.w.ne %s4, %s2, %s0
; CHECK-NEXT:    cmov.w.ne %s5, %s3, %s0
; CHECK-NEXT:    or %s0, 0, %s4
; CHECK-NEXT:    or %s1, 0, %s5
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = select fast i1 %0, fp128 %1, fp128 0xL0000000000000000C000000000000000
  ret fp128 %3
}
