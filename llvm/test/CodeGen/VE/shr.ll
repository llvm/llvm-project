; RUN: llc < %s -mtriple=ve | FileCheck %s

;;; Test both ‘lshr’ and `ashr` instructions
;;;
;;; ‘lshr’ Instruction
;;;
;;; Syntax:
;;;   <result> = lshr <ty> <op1>, <op2>         ; yields ty:result
;;;   <result> = lshr exact <ty> <op1>, <op2>   ; yields ty:result
;;;
;;; Overview:
;;;   The ‘lshr’ instruction (logical shift right) returns the first operand
;;;   shifted to the right a specified number of bits with zero fill.
;;;
;;; Arguments:
;;;   Both arguments to the ‘lshr’ instruction must be the same integer or
;;;   vector of integer type. ‘op2’ is treated as an unsigned value.
;;;
;;; Semantics:
;;;   This instruction always performs a logical shift right operation. The
;;;   most significant bits of the result will be filled with zero bits after
;;;   the shift. If op2 is (statically or dynamically) equal to or larger than
;;;   the number of bits in op1, this instruction returns a poison value. If
;;;   the arguments are vectors, each vector element of op1 is shifted by the
;;;   corresponding shift amount in op2.
;;;
;;;   If the exact keyword is present, the result value of the lshr is a
;;;   poison value if any of the bits shifted out are non-zero.
;;;
;;; Example:
;;;   <result> = lshr i32 4, 1   ; yields i32:result = 2
;;;   <result> = lshr i32 4, 2   ; yields i32:result = 1
;;;   <result> = lshr i8  4, 3   ; yields i8:result = 0
;;;   <result> = lshr i8 -2, 1   ; yields i8:result = 0x7F
;;;   <result> = lshr i32 1, 32  ; undefined
;;;   <result> = lshr <2 x i32> < i32 -2, i32 4>, < i32 1, i32 2>
;;;                          ; yields: result=<2 x i32> < i32 0x7FFFFFFF, i32 1>
;;;
;;; ‘ashr’ Instruction
;;;
;;; Syntax:
;;;   <result> = ashr <ty> <op1>, <op2>         ; yields ty:result
;;;   <result> = ashr exact <ty> <op1>, <op2>   ; yields ty:result
;;;
;;; Overview:
;;;   The ‘ashr’ instruction (arithmetic shift right) returns the first operand
;;;   shifted to the right a specified number of bits with sign extension.
;;;
;;; Arguments:
;;;   Both arguments to the ‘ashr’ instruction must be the same integer or
;;;   vector of integer type. ‘op2’ is treated as an unsigned value.
;;;
;;; Semantics:
;;;   This instruction always performs an arithmetic shift right operation, The
;;;   most significant bits of the result will be filled with the sign bit of
;;;   op1. If op2 is (statically or dynamically) equal to or larger than the
;;;   number of bits in op1, this instruction returns a poison value. If the
;;;   arguments are vectors, each vector element of op1 is shifted by the
;;;   corresponding shift amount in op2.
;;;
;;;   If the exact keyword is present, the result value of the ashr is a poison
;;;   value if any of the bits shifted out are non-zero.
;;;
;;; Example:
;;;   <result> = ashr i32 4, 1   ; yields i32:result = 2
;;;   <result> = ashr i32 4, 2   ; yields i32:result = 1
;;;   <result> = ashr i8  4, 3   ; yields i8:result = 0
;;;   <result> = ashr i8 -2, 1   ; yields i8:result = -1
;;;   <result> = ashr i32 1, 32  ; undefined
;;;   <result> = ashr <2 x i32> < i32 -2, i32 4>, < i32 1, i32 3>
;;;                                  ; yields: result=<2 x i32> < i32 -1, i32 0>
;;;
;;; Note:
;;;   We test only i8/i16/i32/i64/i128 and unsigned of them.

; Function Attrs: norecurse nounwind readnone
define signext i8 @shl_i8_var(i8 signext %0, i8 signext %1) {
; CHECK-LABEL: shl_i8_var:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s1, %s1, (56)0
; CHECK-NEXT:    sra.w.sx %s0, %s0, %s1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = sext i8 %0 to i32
  %4 = zext i8 %1 to i32
  %5 = ashr i32 %3, %4
  %6 = trunc i32 %5 to i8
  ret i8 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i8 @shl_u8_var(i8 zeroext %0, i8 zeroext %1) {
; CHECK-LABEL: shl_u8_var:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    srl %s0, %s0, %s1
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = zext i8 %0 to i32
  %4 = zext i8 %1 to i32
  %5 = lshr i32 %3, %4
  %6 = trunc i32 %5 to i8
  ret i8 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i16 @shl_i16_var(i16 signext %0, i16 signext %1) {
; CHECK-LABEL: shl_i16_var:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s1, %s1, (48)0
; CHECK-NEXT:    sra.w.sx %s0, %s0, %s1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = sext i16 %0 to i32
  %4 = zext i16 %1 to i32
  %5 = ashr i32 %3, %4
  %6 = trunc i32 %5 to i16
  ret i16 %6
}

; Function Attrs: norecurse nounwind readnone
define zeroext i16 @shl_u16_var(i16 zeroext %0, i16 zeroext %1) {
; CHECK-LABEL: shl_u16_var:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    srl %s0, %s0, %s1
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = zext i16 %0 to i32
  %4 = zext i16 %1 to i32
  %5 = lshr i32 %3, %4
  %6 = trunc i32 %5 to i16
  ret i16 %6
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @shl_i32_var(i32 signext %0, i32 signext %1) {
; CHECK-LABEL: shl_i32_var:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sra.w.sx %s0, %s0, %s1
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = ashr i32 %0, %1
  ret i32 %3
}

; Function Attrs: norecurse nounwind readnone
define zeroext i32 @shl_u32_var(i32 zeroext %0, i32 zeroext %1) {
; CHECK-LABEL: shl_u32_var:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    srl %s0, %s0, %s1
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = lshr i32 %0, %1
  ret i32 %3
}

; Function Attrs: norecurse nounwind readnone
define i64 @shl_i64_var(i64 %0, i64 %1) {
; CHECK-LABEL: shl_i64_var:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sra.l %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = ashr i64 %0, %1
  ret i64 %3
}

; Function Attrs: norecurse nounwind readnone
define i64 @shl_u64_var(i64 %0, i64 %1) {
; CHECK-LABEL: shl_u64_var:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    srl %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = lshr i64 %0, %1
  ret i64 %3
}

; Function Attrs: norecurse nounwind readnone
define i128 @shl_i128_var(i128 %0, i128 %1) {
; CHECK-LABEL: shl_i128_var:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s2, %s2, (0)1
; CHECK-NEXT:    lea %s3, __ashrti3@lo
; CHECK-NEXT:    and %s3, %s3, (32)0
; CHECK-NEXT:    lea.sl %s12, __ashrti3@hi(, %s3)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = ashr i128 %0, %1
  ret i128 %3
}

; Function Attrs: norecurse nounwind readnone
define i128 @shl_u128_var(i128 %0, i128 %1) {
; CHECK-LABEL: shl_u128_var:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea %s3, __lshrti3@lo
; CHECK-NEXT:    and %s3, %s3, (32)0
; CHECK-NEXT:    lea.sl %s12, __lshrti3@hi(, %s3)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = lshr i128 %0, %1
  ret i128 %3
}

; Function Attrs: norecurse nounwind readnone
define signext i8 @shl_const_i8(i8 signext %0) {
; CHECK-LABEL: shl_const_i8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    sra.w.sx %s0, (62)1, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = zext i8 %0 to i32
  %3 = ashr i32 -4, %2
  %4 = trunc i32 %3 to i8
  ret i8 %4
}

; Function Attrs: norecurse nounwind readnone
define zeroext i8 @shl_const_u8(i8 zeroext %0) {
; CHECK-LABEL: shl_const_u8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sra.w.sx %s0, (62)1, %s0
; CHECK-NEXT:    and %s0, %s0, (56)0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = zext i8 %0 to i32
  %3 = ashr i32 -4, %2
  %4 = trunc i32 %3 to i8
  ret i8 %4
}

; Function Attrs: norecurse nounwind readnone
define signext i16 @shl_const_i16(i16 signext %0) {
; CHECK-LABEL: shl_const_i16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s0, %s0, (48)0
; CHECK-NEXT:    sra.w.sx %s0, (62)1, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = zext i16 %0 to i32
  %3 = ashr i32 -4, %2
  %4 = trunc i32 %3 to i16
  ret i16 %4
}

; Function Attrs: norecurse nounwind readnone
define zeroext i16 @shl_const_u16(i16 zeroext %0) {
; CHECK-LABEL: shl_const_u16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sra.w.sx %s0, (62)1, %s0
; CHECK-NEXT:    and %s0, %s0, (48)0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = zext i16 %0 to i32
  %3 = ashr i32 -4, %2
  %4 = trunc i32 %3 to i16
  ret i16 %4
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @shl_const_i32(i32 signext %0) {
; CHECK-LABEL: shl_const_i32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sra.w.sx %s0, (62)1, %s0
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = ashr i32 -4, %0
  ret i32 %2
}

; Function Attrs: norecurse nounwind readnone
define zeroext i32 @shl_const_u32(i32 zeroext %0) {
; CHECK-LABEL: shl_const_u32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sra.w.sx %s0, (62)1, %s0
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = ashr i32 -4, %0
  ret i32 %2
}

; Function Attrs: norecurse nounwind readnone
define i64 @shl_const_i64(i64 %0) {
; CHECK-LABEL: shl_const_i64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sra.l %s0, (62)1, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = ashr i64 -4, %0
  ret i64 %2
}

; Function Attrs: norecurse nounwind readnone
define i64 @shl_const_u64(i64 %0) {
; CHECK-LABEL: shl_const_u64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sra.l %s0, (62)1, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = ashr i64 -4, %0
  ret i64 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @shl_const_i128(i128 %0) {
; CHECK-LABEL: shl_const_i128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s2, %s0, (0)1
; CHECK-NEXT:    lea %s0, __ashrti3@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __ashrti3@hi(, %s0)
; CHECK-NEXT:    or %s0, -4, (0)1
; CHECK-NEXT:    or %s1, -1, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = ashr i128 -4, %0
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @shl_const_u128(i128 %0) {
; CHECK-LABEL: shl_const_u128:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    adds.w.sx %s2, %s0, (0)1
; CHECK-NEXT:    lea %s0, __ashrti3@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __ashrti3@hi(, %s0)
; CHECK-NEXT:    or %s0, -4, (0)1
; CHECK-NEXT:    or %s1, -1, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = ashr i128 -4, %0
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define signext i8 @shl_i8_const(i8 signext %0) {
; CHECK-LABEL: shl_i8_const:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sra.w.sx %s0, %s0, 3
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = ashr i8 %0, 3
  ret i8 %2
}

; Function Attrs: norecurse nounwind readnone
define zeroext i8 @shl_u8_const(i8 zeroext %0) {
; CHECK-LABEL: shl_u8_const:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    srl %s0, %s0, 3
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = lshr i8 %0, 3
  ret i8 %2
}

; Function Attrs: norecurse nounwind readnone
define signext i16 @shl_i16_const(i16 signext %0) {
; CHECK-LABEL: shl_i16_const:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sra.w.sx %s0, %s0, 7
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = ashr i16 %0, 7
  ret i16 %2
}

; Function Attrs: norecurse nounwind readnone
define zeroext i16 @shl_u16_const(i16 zeroext %0) {
; CHECK-LABEL: shl_u16_const:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    srl %s0, %s0, 7
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = lshr i16 %0, 7
  ret i16 %2
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @shl_i32_const(i32 signext %0) {
; CHECK-LABEL: shl_i32_const:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sra.w.sx %s0, %s0, 15
; CHECK-NEXT:    adds.w.sx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = ashr i32 %0, 15
  ret i32 %2
}

; Function Attrs: norecurse nounwind readnone
define zeroext i32 @shl_u32_const(i32 zeroext %0) {
; CHECK-LABEL: shl_u32_const:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    srl %s0, %s0, 15
; CHECK-NEXT:    adds.w.zx %s0, %s0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = lshr i32 %0, 15
  ret i32 %2
}

; Function Attrs: norecurse nounwind readnone
define i64 @shl_i64_const(i64 %0) {
; CHECK-LABEL: shl_i64_const:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sra.l %s0, %s0, 63
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = ashr i64 %0, 63
  ret i64 %2
}

; Function Attrs: norecurse nounwind readnone
define i64 @shl_u64_const(i64 %0) {
; CHECK-LABEL: shl_u64_const:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    srl %s0, %s0, 63
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = lshr i64 %0, 63
  ret i64 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @shl_i128_const(i128 %0) {
; CHECK-LABEL: shl_i128_const:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    sra.l %s0, %s1, 63
; CHECK-NEXT:    or %s1, 0, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = ashr i128 %0, 127
  ret i128 %2
}

; Function Attrs: norecurse nounwind readnone
define i128 @shl_u128_const(i128 %0) {
; CHECK-LABEL: shl_u128_const:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    srl %s0, %s1, 63
; CHECK-NEXT:    or %s1, 0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = lshr i128 %0, 127
  ret i128 %2
}
