; Test insertions of 32-bit constants into one half of an i64.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Prefer LHI over IILF for signed 16-bit constants.
define i64 @f1(i64 %a) {
; CHECK-LABEL: f1:
; CHECK-NOT: ni
; CHECK: lhi %r2, 1
; CHECK: br %r14
  %and = and i64 %a, 18446744069414584320
  %or = or i64 %and, 1
  ret i64 %or
}

; Check the high end of the LHI range.
define i64 @f2(i64 %a) {
; CHECK-LABEL: f2:
; CHECK-NOT: ni
; CHECK: lhi %r2, 32767
; CHECK: br %r14
  %and = and i64 %a, 18446744069414584320
  %or = or i64 %and, 32767
  ret i64 %or
}

; Check the next value up, which should use IILF instead.
define i64 @f3(i64 %a) {
; CHECK-LABEL: f3:
; CHECK-NOT: ni
; CHECK: iilf %r2, 32768
; CHECK: br %r14
  %and = and i64 %a, 18446744069414584320
  %or = or i64 %and, 32768
  ret i64 %or
}

; Check a value in which the lower 16 bits are clear.
define i64 @f4(i64 %a) {
; CHECK-LABEL: f4:
; CHECK-NOT: ni
; CHECK: iilf %r2, 65536
; CHECK: br %r14
  %and = and i64 %a, 18446744069414584320
  %or = or i64 %and, 65536
  ret i64 %or
}

; Check the highest useful IILF value (-0x8001).
define i64 @f5(i64 %a) {
; CHECK-LABEL: f5:
; CHECK-NOT: ni
; CHECK: iilf %r2, 4294934527
; CHECK: br %r14
  %and = and i64 %a, 18446744069414584320
  %or = or i64 %and, 4294934527
  ret i64 %or
}

; Check the next value up, which should use LHI instead.
define i64 @f6(i64 %a) {
; CHECK-LABEL: f6:
; CHECK-NOT: ni
; CHECK: lhi %r2, -32768
; CHECK: br %r14
  %and = and i64 %a, 18446744069414584320
  %or = or i64 %and, 4294934528
  ret i64 %or
}

; Check the highest useful LHI value.  (We use OILF for -1 instead, although
; LHI might be better there too.)
define i64 @f7(i64 %a) {
; CHECK-LABEL: f7:
; CHECK-NOT: ni
; CHECK: lhi %r2, -2
; CHECK: br %r14
  %and = and i64 %a, 18446744069414584320
  %or = or i64 %and, 4294967294
  ret i64 %or
}

; Check that SRLG is still used if some of the high bits are known to be 0
; (and so might be removed from the mask).
define i64 @f8(i64 %a) {
; CHECK-LABEL: f8:
; CHECK: srlg %r2, %r2, 1
; CHECK-NEXT: iilf %r2, 32768
; CHECK: br %r14
  %shifted = lshr i64 %a, 1
  %and = and i64 %shifted, 18446744069414584320
  %or = or i64 %and, 32768
  ret i64 %or
}

; Repeat f8 with addition, which is known to be equivalent to OR in this case.
define i64 @f9(i64 %a) {
; CHECK-LABEL: f9:
; CHECK: srlg %r2, %r2, 1
; CHECK-NEXT: iilf %r2, 32768
; CHECK: br %r14
  %shifted = lshr i64 %a, 1
  %and = and i64 %shifted, 18446744069414584320
  %or = add i64 %and, 32768
  ret i64 %or
}

; Repeat f8 with already-zero bits removed from the mask.
define i64 @f10(i64 %a) {
; CHECK-LABEL: f10:
; CHECK: srlg %r2, %r2, 1
; CHECK-NEXT: iilf %r2, 32768
; CHECK: br %r14
  %shifted = lshr i64 %a, 1
  %and = and i64 %shifted, 9223372032559808512
  %or = or i64 %and, 32768
  ret i64 %or
}

; Repeat f10 with addition, which is known to be equivalent to OR in this case.
define i64 @f11(i64 %a) {
; CHECK-LABEL: f11:
; CHECK: srlg %r2, %r2, 1
; CHECK-NEXT: iilf %r2, 32768
; CHECK: br %r14
  %shifted = lshr i64 %a, 1
  %and = and i64 %shifted, 9223372032559808512
  %or = add i64 %and, 32768
  ret i64 %or
}

; Check the lowest useful IIHF value.
define i64 @f12(i64 %a) {
; CHECK-LABEL: f12:
; CHECK-NOT: ni
; CHECK: iihf %r2, 1
; CHECK: br %r14
  %and = and i64 %a, 4294967295
  %or = or i64 %and, 4294967296
  ret i64 %or
}

; Check a value in which the lower 16 bits are clear.
define i64 @f13(i64 %a) {
; CHECK-LABEL: f13:
; CHECK-NOT: ni
; CHECK: iihf %r2, 2147483648
; CHECK: br %r14
  %and = and i64 %a, 4294967295
  %or = or i64 %and, 9223372036854775808
  ret i64 %or
}

; Check the highest useful IIHF value (0xfffffffe).
define i64 @f14(i64 %a) {
; CHECK-LABEL: f14:
; CHECK-NOT: ni
; CHECK: iihf %r2, 4294967294
; CHECK: br %r14
  %and = and i64 %a, 4294967295
  %or = or i64 %and, 18446744065119617024
  ret i64 %or
}

; Check a case in which some of the low 32 bits are known to be clear,
; and so could be removed from the AND mask.
define i64 @f15(i64 %a) {
; CHECK-LABEL: f15:
; CHECK: sllg %r2, %r2, 1
; CHECK-NEXT: iihf %r2, 1
; CHECK: br %r14
  %shifted = shl i64 %a, 1
  %and = and i64 %shifted, 4294967295
  %or = or i64 %and, 4294967296
  ret i64 %or
}

; Repeat f15 with the zero bits explicitly removed from the mask.
define i64 @f16(i64 %a) {
; CHECK-LABEL: f16:
; CHECK: sllg %r2, %r2, 1
; CHECK-NEXT: iihf %r2, 1
; CHECK: br %r14
  %shifted = shl i64 %a, 1
  %and = and i64 %shifted, 4294967294
  %or = or i64 %and, 4294967296
  ret i64 %or
}

; Check concatenation of two i32s.
define i64 @f17(i32 %a) {
; CHECK-LABEL: f17:
; CHECK: msr %r2, %r2
; CHECK-NEXT: iihf %r2, 1
; CHECK: br %r14
  %mul = mul i32 %a, %a
  %ext = zext i32 %mul to i64
  %or = or i64 %ext, 4294967296
  ret i64 %or
}

; Repeat f17 with the operands reversed.
define i64 @f18(i32 %a) {
; CHECK-LABEL: f18:
; CHECK: msr %r2, %r2
; CHECK-NEXT: iihf %r2, 1
; CHECK: br %r14
  %mul = mul i32 %a, %a
  %ext = zext i32 %mul to i64
  %or = or i64 4294967296, %ext
  ret i64 %or
}

; The truncation here isn't free; we need an explicit zero extension.
define i64 @f19(i32 %a) {
; CHECK-LABEL: f19:
; CHECK: llgcr %r2, %r2
; CHECK: oihl %r2, 1
; CHECK: br %r14
  %trunc = trunc i32 %a to i8
  %ext = zext i8 %trunc to i64
  %or = or i64 %ext, 4294967296
  ret i64 %or
}
