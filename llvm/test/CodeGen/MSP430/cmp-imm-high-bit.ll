; RUN: llc -mtriple=msp430 < %s | FileCheck %s

; Regression tests for EmitCMP folding of LHS constants whose i16 high bit is
; set. These exercise the four EmitCMP branches that turn `c CMP rhs` into
; `rhs CMP' c+1`. The constant addition must use APInt arithmetic at the
; original bit width; computing it via getSExtValue() and then handing the
; resulting int64_t to getConstant(uint64_t, ..., MVT::i16) trips the
; isUIntN(16, val) assertion in the APInt constructor for any constant with
; bit 15 set.

; CHECK-LABEL: cmp_ule_high_bit:
; Folded into `rhs u< 0x8001` (c = 0x8000, c+1 = 0x8001 = -32767 signed).
; CHECK: cmp #-32767, r12
define i16 @cmp_ule_high_bit(i16 %a) nounwind {
  %t = icmp ule i16 %a, 32768
  %r = zext i1 %t to i16
  ret i16 %r
}

; CHECK-LABEL: cmp_ugt_high_bit:
; Folded into `rhs u>= 0x8001`.
; CHECK: cmp #-32767, r12
define i16 @cmp_ugt_high_bit(i16 %a) nounwind {
  %t = icmp ugt i16 %a, 32768
  %r = zext i1 %t to i16
  ret i16 %r
}

; CHECK-LABEL: cmp_sle_neg_high_bit:
; Folded into `rhs s< -32766` (c = -32767, c+1 = -32766).
; CHECK: cmp #-32766,
define i16 @cmp_sle_neg_high_bit(i16 %a) nounwind {
  %t = icmp sle i16 %a, -32767
  %r = zext i1 %t to i16
  ret i16 %r
}

; CHECK-LABEL: cmp_sgt_neg_high_bit:
; Folded into `rhs s>= -32766`.
; CHECK: cmp #-32766,
define i16 @cmp_sgt_neg_high_bit(i16 %a) nounwind {
  %t = icmp sgt i16 %a, -32767
  %r = zext i1 %t to i16
  ret i16 %r
}

; The br_cc path goes through the same EmitCMP helper.
; CHECK-LABEL: br_ule_high_bit:
; CHECK: cmp #-32767, r12
define i16 @br_ule_high_bit(i16 %a) nounwind {
  %t = icmp ule i16 %a, 32768
  br i1 %t, label %yes, label %no
yes:
  ret i16 1
no:
  ret i16 0
}

; CHECK-LABEL: br_sle_neg_high_bit:
; CHECK: cmp #-32766, r12
define i16 @br_sle_neg_high_bit(i16 %a) nounwind {
  %t = icmp sle i16 %a, -32767
  br i1 %t, label %yes, label %no
yes:
  ret i16 1
no:
  ret i16 0
}
