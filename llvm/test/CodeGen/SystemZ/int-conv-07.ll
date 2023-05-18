; Test sign extensions from a halfword to an i64.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test register extension, starting with an i32.
define i64 @f1(i64 %a) {
; CHECK-LABEL: f1:
; CHECK: lghr %r2, %r2
; CHECK: br %r14
  %half = trunc i64 %a to i16
  %ext = sext i16 %half to i64
  ret i64 %ext
}

; ...and again with an i64.
define i64 @f2(i32 %a) {
; CHECK-LABEL: f2:
; CHECK: lghr %r2, %r2
; CHECK: br %r14
  %half = trunc i32 %a to i16
  %ext = sext i16 %half to i64
  ret i64 %ext
}

; Check LGH with no displacement.
define i64 @f3(ptr %src) {
; CHECK-LABEL: f3:
; CHECK: lgh %r2, 0(%r2)
; CHECK: br %r14
  %half = load i16, ptr %src
  %ext = sext i16 %half to i64
  ret i64 %ext
}

; Check the high end of the LGH range.
define i64 @f4(ptr %src) {
; CHECK-LABEL: f4:
; CHECK: lgh %r2, 524286(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %src, i64 262143
  %half = load i16, ptr %ptr
  %ext = sext i16 %half to i64
  ret i64 %ext
}

; Check the next halfword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f5(ptr %src) {
; CHECK-LABEL: f5:
; CHECK: agfi %r2, 524288
; CHECK: lgh %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %src, i64 262144
  %half = load i16, ptr %ptr
  %ext = sext i16 %half to i64
  ret i64 %ext
}

; Check the high end of the negative LGH range.
define i64 @f6(ptr %src) {
; CHECK-LABEL: f6:
; CHECK: lgh %r2, -2(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %src, i64 -1
  %half = load i16, ptr %ptr
  %ext = sext i16 %half to i64
  ret i64 %ext
}

; Check the low end of the LGH range.
define i64 @f7(ptr %src) {
; CHECK-LABEL: f7:
; CHECK: lgh %r2, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %src, i64 -262144
  %half = load i16, ptr %ptr
  %ext = sext i16 %half to i64
  ret i64 %ext
}

; Check the next halfword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f8(ptr %src) {
; CHECK-LABEL: f8:
; CHECK: agfi %r2, -524290
; CHECK: lgh %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %src, i64 -262145
  %half = load i16, ptr %ptr
  %ext = sext i16 %half to i64
  ret i64 %ext
}

; Check that LGH allows an index.
define i64 @f9(i64 %src, i64 %index) {
; CHECK-LABEL: f9:
; CHECK: lgh %r2, 524287(%r3,%r2)
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to ptr
  %half = load i16, ptr %ptr
  %ext = sext i16 %half to i64
  ret i64 %ext
}

; Test a case where we spill the source of at least one LGHR.  We want
; to use LGH if possible.
define void @f10(ptr %ptr) {
; CHECK-LABEL: f10:
; CHECK: lgh {{%r[0-9]+}}, 198(%r15)
; CHECK: br %r14
  %val0 = load volatile i64, ptr %ptr
  %val1 = load volatile i64, ptr %ptr
  %val2 = load volatile i64, ptr %ptr
  %val3 = load volatile i64, ptr %ptr
  %val4 = load volatile i64, ptr %ptr
  %val5 = load volatile i64, ptr %ptr
  %val6 = load volatile i64, ptr %ptr
  %val7 = load volatile i64, ptr %ptr
  %val8 = load volatile i64, ptr %ptr
  %val9 = load volatile i64, ptr %ptr
  %val10 = load volatile i64, ptr %ptr
  %val11 = load volatile i64, ptr %ptr
  %val12 = load volatile i64, ptr %ptr
  %val13 = load volatile i64, ptr %ptr
  %val14 = load volatile i64, ptr %ptr
  %val15 = load volatile i64, ptr %ptr

  %trunc0 = trunc i64 %val0 to i16
  %trunc1 = trunc i64 %val1 to i16
  %trunc2 = trunc i64 %val2 to i16
  %trunc3 = trunc i64 %val3 to i16
  %trunc4 = trunc i64 %val4 to i16
  %trunc5 = trunc i64 %val5 to i16
  %trunc6 = trunc i64 %val6 to i16
  %trunc7 = trunc i64 %val7 to i16
  %trunc8 = trunc i64 %val8 to i16
  %trunc9 = trunc i64 %val9 to i16
  %trunc10 = trunc i64 %val10 to i16
  %trunc11 = trunc i64 %val11 to i16
  %trunc12 = trunc i64 %val12 to i16
  %trunc13 = trunc i64 %val13 to i16
  %trunc14 = trunc i64 %val14 to i16
  %trunc15 = trunc i64 %val15 to i16

  %ext0 = sext i16 %trunc0 to i64
  %ext1 = sext i16 %trunc1 to i64
  %ext2 = sext i16 %trunc2 to i64
  %ext3 = sext i16 %trunc3 to i64
  %ext4 = sext i16 %trunc4 to i64
  %ext5 = sext i16 %trunc5 to i64
  %ext6 = sext i16 %trunc6 to i64
  %ext7 = sext i16 %trunc7 to i64
  %ext8 = sext i16 %trunc8 to i64
  %ext9 = sext i16 %trunc9 to i64
  %ext10 = sext i16 %trunc10 to i64
  %ext11 = sext i16 %trunc11 to i64
  %ext12 = sext i16 %trunc12 to i64
  %ext13 = sext i16 %trunc13 to i64
  %ext14 = sext i16 %trunc14 to i64
  %ext15 = sext i16 %trunc15 to i64

  store volatile i64 %val0, ptr %ptr
  store volatile i64 %val1, ptr %ptr
  store volatile i64 %val2, ptr %ptr
  store volatile i64 %val3, ptr %ptr
  store volatile i64 %val4, ptr %ptr
  store volatile i64 %val5, ptr %ptr
  store volatile i64 %val6, ptr %ptr
  store volatile i64 %val7, ptr %ptr
  store volatile i64 %val8, ptr %ptr
  store volatile i64 %val9, ptr %ptr
  store volatile i64 %val10, ptr %ptr
  store volatile i64 %val11, ptr %ptr
  store volatile i64 %val12, ptr %ptr
  store volatile i64 %val13, ptr %ptr
  store volatile i64 %val14, ptr %ptr
  store volatile i64 %val15, ptr %ptr

  store volatile i64 %ext0, ptr %ptr
  store volatile i64 %ext1, ptr %ptr
  store volatile i64 %ext2, ptr %ptr
  store volatile i64 %ext3, ptr %ptr
  store volatile i64 %ext4, ptr %ptr
  store volatile i64 %ext5, ptr %ptr
  store volatile i64 %ext6, ptr %ptr
  store volatile i64 %ext7, ptr %ptr
  store volatile i64 %ext8, ptr %ptr
  store volatile i64 %ext9, ptr %ptr
  store volatile i64 %ext10, ptr %ptr
  store volatile i64 %ext11, ptr %ptr
  store volatile i64 %ext12, ptr %ptr
  store volatile i64 %ext13, ptr %ptr
  store volatile i64 %ext14, ptr %ptr
  store volatile i64 %ext15, ptr %ptr

  ret void
}
