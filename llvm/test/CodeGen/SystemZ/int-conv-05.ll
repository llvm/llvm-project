; Test sign extensions from a halfword to an i32.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test register extension, starting with an i32.
define i32 @f1(i32 %a) {
; CHECK-LABEL: f1:
; CHECK: lhr %r2, %r2
; CHECK: br %r14
  %half = trunc i32 %a to i16
  %ext = sext i16 %half to i32
  ret i32 %ext
}

; ...and again with an i64.
define i32 @f2(i64 %a) {
; CHECK-LABEL: f2:
; CHECK: lhr %r2, %r2
; CHECK: br %r14
  %half = trunc i64 %a to i16
  %ext = sext i16 %half to i32
  ret i32 %ext
}

; Check the low end of the LH range.
define i32 @f3(ptr %src) {
; CHECK-LABEL: f3:
; CHECK: lh %r2, 0(%r2)
; CHECK: br %r14
  %half = load i16, ptr %src
  %ext = sext i16 %half to i32
  ret i32 %ext
}

; Check the high end of the LH range.
define i32 @f4(ptr %src) {
; CHECK-LABEL: f4:
; CHECK: lh %r2, 4094(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %src, i64 2047
  %half = load i16, ptr %ptr
  %ext = sext i16 %half to i32
  ret i32 %ext
}

; Check the next halfword up, which needs LHY rather than LH.
define i32 @f5(ptr %src) {
; CHECK-LABEL: f5:
; CHECK: lhy %r2, 4096(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %src, i64 2048
  %half = load i16, ptr %ptr
  %ext = sext i16 %half to i32
  ret i32 %ext
}

; Check the high end of the LHY range.
define i32 @f6(ptr %src) {
; CHECK-LABEL: f6:
; CHECK: lhy %r2, 524286(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %src, i64 262143
  %half = load i16, ptr %ptr
  %ext = sext i16 %half to i32
  ret i32 %ext
}

; Check the next halfword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f7(ptr %src) {
; CHECK-LABEL: f7:
; CHECK: agfi %r2, 524288
; CHECK: lh %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %src, i64 262144
  %half = load i16, ptr %ptr
  %ext = sext i16 %half to i32
  ret i32 %ext
}

; Check the high end of the negative LHY range.
define i32 @f8(ptr %src) {
; CHECK-LABEL: f8:
; CHECK: lhy %r2, -2(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %src, i64 -1
  %half = load i16, ptr %ptr
  %ext = sext i16 %half to i32
  ret i32 %ext
}

; Check the low end of the LHY range.
define i32 @f9(ptr %src) {
; CHECK-LABEL: f9:
; CHECK: lhy %r2, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %src, i64 -262144
  %half = load i16, ptr %ptr
  %ext = sext i16 %half to i32
  ret i32 %ext
}

; Check the next halfword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f10(ptr %src) {
; CHECK-LABEL: f10:
; CHECK: agfi %r2, -524290
; CHECK: lh %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %src, i64 -262145
  %half = load i16, ptr %ptr
  %ext = sext i16 %half to i32
  ret i32 %ext
}

; Check that LH allows an index
define i32 @f11(i64 %src, i64 %index) {
; CHECK-LABEL: f11:
; CHECK: lh %r2, 4094(%r3,%r2)
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4094
  %ptr = inttoptr i64 %add2 to ptr
  %half = load i16, ptr %ptr
  %ext = sext i16 %half to i32
  ret i32 %ext
}

; Check that LH allows an index
define i32 @f12(i64 %src, i64 %index) {
; CHECK-LABEL: f12:
; CHECK: lhy %r2, 4096(%r3,%r2)
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4096
  %ptr = inttoptr i64 %add2 to ptr
  %half = load i16, ptr %ptr
  %ext = sext i16 %half to i32
  ret i32 %ext
}

; Test a case where we spill the source of at least one LHR.  We want
; to use LH if possible.
define void @f13(ptr %ptr) {
; CHECK-LABEL: f13:
; CHECK: lh {{%r[0-9]+}}, 16{{[26]}}(%r15)
; CHECK: br %r14
  %val0 = load volatile i32, ptr %ptr
  %val1 = load volatile i32, ptr %ptr
  %val2 = load volatile i32, ptr %ptr
  %val3 = load volatile i32, ptr %ptr
  %val4 = load volatile i32, ptr %ptr
  %val5 = load volatile i32, ptr %ptr
  %val6 = load volatile i32, ptr %ptr
  %val7 = load volatile i32, ptr %ptr
  %val8 = load volatile i32, ptr %ptr
  %val9 = load volatile i32, ptr %ptr
  %val10 = load volatile i32, ptr %ptr
  %val11 = load volatile i32, ptr %ptr
  %val12 = load volatile i32, ptr %ptr
  %val13 = load volatile i32, ptr %ptr
  %val14 = load volatile i32, ptr %ptr
  %val15 = load volatile i32, ptr %ptr

  %trunc0 = trunc i32 %val0 to i16
  %trunc1 = trunc i32 %val1 to i16
  %trunc2 = trunc i32 %val2 to i16
  %trunc3 = trunc i32 %val3 to i16
  %trunc4 = trunc i32 %val4 to i16
  %trunc5 = trunc i32 %val5 to i16
  %trunc6 = trunc i32 %val6 to i16
  %trunc7 = trunc i32 %val7 to i16
  %trunc8 = trunc i32 %val8 to i16
  %trunc9 = trunc i32 %val9 to i16
  %trunc10 = trunc i32 %val10 to i16
  %trunc11 = trunc i32 %val11 to i16
  %trunc12 = trunc i32 %val12 to i16
  %trunc13 = trunc i32 %val13 to i16
  %trunc14 = trunc i32 %val14 to i16
  %trunc15 = trunc i32 %val15 to i16

  %ext0 = sext i16 %trunc0 to i32
  %ext1 = sext i16 %trunc1 to i32
  %ext2 = sext i16 %trunc2 to i32
  %ext3 = sext i16 %trunc3 to i32
  %ext4 = sext i16 %trunc4 to i32
  %ext5 = sext i16 %trunc5 to i32
  %ext6 = sext i16 %trunc6 to i32
  %ext7 = sext i16 %trunc7 to i32
  %ext8 = sext i16 %trunc8 to i32
  %ext9 = sext i16 %trunc9 to i32
  %ext10 = sext i16 %trunc10 to i32
  %ext11 = sext i16 %trunc11 to i32
  %ext12 = sext i16 %trunc12 to i32
  %ext13 = sext i16 %trunc13 to i32
  %ext14 = sext i16 %trunc14 to i32
  %ext15 = sext i16 %trunc15 to i32

  store volatile i32 %val0, ptr %ptr
  store volatile i32 %val1, ptr %ptr
  store volatile i32 %val2, ptr %ptr
  store volatile i32 %val3, ptr %ptr
  store volatile i32 %val4, ptr %ptr
  store volatile i32 %val5, ptr %ptr
  store volatile i32 %val6, ptr %ptr
  store volatile i32 %val7, ptr %ptr
  store volatile i32 %val8, ptr %ptr
  store volatile i32 %val9, ptr %ptr
  store volatile i32 %val10, ptr %ptr
  store volatile i32 %val11, ptr %ptr
  store volatile i32 %val12, ptr %ptr
  store volatile i32 %val13, ptr %ptr
  store volatile i32 %val14, ptr %ptr
  store volatile i32 %val15, ptr %ptr

  store volatile i32 %ext0, ptr %ptr
  store volatile i32 %ext1, ptr %ptr
  store volatile i32 %ext2, ptr %ptr
  store volatile i32 %ext3, ptr %ptr
  store volatile i32 %ext4, ptr %ptr
  store volatile i32 %ext5, ptr %ptr
  store volatile i32 %ext6, ptr %ptr
  store volatile i32 %ext7, ptr %ptr
  store volatile i32 %ext8, ptr %ptr
  store volatile i32 %ext9, ptr %ptr
  store volatile i32 %ext10, ptr %ptr
  store volatile i32 %ext11, ptr %ptr
  store volatile i32 %ext12, ptr %ptr
  store volatile i32 %ext13, ptr %ptr
  store volatile i32 %ext14, ptr %ptr
  store volatile i32 %ext15, ptr %ptr

  ret void
}
