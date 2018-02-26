; RUN: llc -O0 -fast-isel -fast-isel-abort=1 -verify-machineinstrs -mtriple=arm64-apple-darwin < %s | FileCheck %s

define i32 @icmp_eq_imm(i32 %a) nounwind ssp {
entry:
; CHECK-LABEL: icmp_eq_imm
; CHECK:       cmp w0, #31
; CHECK-NEXT:  cset w0, eq
  %cmp = icmp eq i32 %a, 31
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @icmp_eq_neg_imm(i32 %a) nounwind ssp {
entry:
; CHECK-LABEL: icmp_eq_neg_imm
; CHECK:       cmn w0, #7
; CHECK-NEXT:  cset w0, eq
  %cmp = icmp eq i32 %a, -7
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @icmp_eq_i32(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK-LABEL: icmp_eq_i32
; CHECK:       cmp w0, w1
; CHECK-NEXT:  cset w0, eq
  %cmp = icmp eq i32 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @icmp_ne(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK-LABEL: icmp_ne
; CHECK:       cmp w0, w1
; CHECK-NEXT:  cset w0, ne
  %cmp = icmp ne i32 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @icmp_eq_ptr(i8* %a) {
entry:
; CHECK-LABEL: icmp_eq_ptr
; CHECK:       cmp x0, #0
; CHECK-NEXT:  cset {{.+}}, eq
  %cmp = icmp eq i8* %a, null
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @icmp_ne_ptr(i8* %a) {
entry:
; CHECK-LABEL: icmp_ne_ptr
; CHECK:       cmp x0, #0
; CHECK-NEXT:  cset {{.+}}, ne
  %cmp = icmp ne i8* %a, null
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @icmp_ugt(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK-LABEL: icmp_ugt
; CHECK:       cmp w0, w1
; CHECK-NEXT:  cset w0, hi
  %cmp = icmp ugt i32 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @icmp_uge(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK-LABEL: icmp_uge
; CHECK:       cmp w0, w1
; CHECK-NEXT:  cset w0, hs
  %cmp = icmp uge i32 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @icmp_ult(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK-LABEL: icmp_ult
; CHECK:       cmp w0, w1
; CHECK-NEXT:  cset w0, lo
  %cmp = icmp ult i32 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @icmp_ule(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK-LABEL: icmp_ule
; CHECK:       cmp w0, w1
; CHECK-NEXT:  cset w0, ls
  %cmp = icmp ule i32 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @icmp_sgt(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK-LABEL: icmp_sgt
; CHECK:       cmp w0, w1
; CHECK-NEXT:  cset w0, gt
  %cmp = icmp sgt i32 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @icmp_sge(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK-LABEL: icmp_sge
; CHECK:       cmp w0, w1
; CHECK-NEXT:  cset w0, ge
  %cmp = icmp sge i32 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @icmp_slt(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK-LABEL: icmp_slt
; CHECK:       cmp w0, w1
; CHECK-NEXT:  cset w0, lt
  %cmp = icmp slt i32 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @icmp_sle(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK-LABEL: icmp_sle
; CHECK:       cmp w0, w1
; CHECK-NEXT:  cset w0, le
  %cmp = icmp sle i32 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @icmp_i64(i64 %a, i64 %b) nounwind ssp {
entry:
; CHECK-LABEL: icmp_i64
; CHECK:       cmp  x0, x1
; CHECK-NEXT:  cset w{{[0-9]+}}, le
  %cmp = icmp sle i64 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define zeroext i1 @icmp_eq_i16(i16 %a, i16 %b) nounwind ssp {
entry:
; CHECK-LABEL: icmp_eq_i16
; CHECK:       sxth w0, w0
; CHECK:       cmp w0, w1, sxth
; CHECK-NEXT:  cset w0, eq
  %cmp = icmp eq i16 %a, %b
  ret i1 %cmp
}

define zeroext i1 @icmp_eq_i8(i8 %a, i8 %b) nounwind ssp {
entry:
; CHECK-LABEL: icmp_eq_i8
; CHECK:       sxtb w0, w0
; CHECK-NEXT:  cmp w0, w1, sxtb
; CHECK-NEXT:  cset w0, eq
  %cmp = icmp eq i8 %a, %b
  ret i1 %cmp
}

define i32 @icmp_i16_unsigned(i16 %a, i16 %b) nounwind {
entry:
; CHECK-LABEL: icmp_i16_unsigned
; CHECK:       uxth w0, w0
; CHECK-NEXT:  cmp w0, w1, uxth
; CHECK-NEXT:  cset w0, lo
  %cmp = icmp ult i16 %a, %b
  %conv2 = zext i1 %cmp to i32
  ret i32 %conv2
}

define i32 @icmp_i8_signed(i8 %a, i8 %b) nounwind {
entry:
; CHECK-LABEL: icmp_i8_signed
; CHECK:       sxtb w0, w0
; CHECK-NEXT:  cmp w0, w1, sxtb
; CHECK-NEXT:  cset w0, gt
  %cmp = icmp sgt i8 %a, %b
  %conv2 = zext i1 %cmp to i32
  ret i32 %conv2
}

define i32 @icmp_i1_signed(i1 %a, i1 %b) nounwind {
entry:
; CHECK-LABEL: icmp_i1_signed
; CHECK:       sbfx [[REG1:w[0-9]+]], w0, #0, #1
; CHECK-NEXT:  sbfx [[REG2:w[0-9]+]], w1, #0, #1
; CHECK-NEXT:  cmp  [[REG1]], [[REG2]]
; CHECK-NEXT:  cset w0, gt
  %cmp = icmp sgt i1 %a, %b
  %conv2 = zext i1 %cmp to i32
  ret i32 %conv2
}

define i32 @icmp_i16_signed_const(i16 %a) nounwind {
entry:
; CHECK-LABEL: icmp_i16_signed_const
; CHECK:       sxth w0, w0
; CHECK-NEXT:  cmn w0, #233
; CHECK-NEXT:  cset w0, lt
; CHECK-NEXT:  and w0, w0, #0x1
  %cmp = icmp slt i16 %a, -233
  %conv2 = zext i1 %cmp to i32
  ret i32 %conv2
}

define i32 @icmp_i8_signed_const(i8 %a) nounwind {
entry:
; CHECK-LABEL: icmp_i8_signed_const
; CHECK:       sxtb w0, w0
; CHECK-NEXT:  cmp w0, #124
; CHECK-NEXT:  cset w0, gt
; CHECK-NEXT:  and w0, w0, #0x1
  %cmp = icmp sgt i8 %a, 124
  %conv2 = zext i1 %cmp to i32
  ret i32 %conv2
}

define i32 @icmp_i1_unsigned_const(i1 %a) nounwind {
entry:
; CHECK-LABEL: icmp_i1_unsigned_const
; CHECK:       and w0, w0, #0x1
; CHECK-NEXT:  cmp w0, #0
; CHECK-NEXT:  cset w0, lo
; CHECK-NEXT:  and w0, w0, #0x1
  %cmp = icmp ult i1 %a, 0
  %conv2 = zext i1 %cmp to i32
  ret i32 %conv2
}
