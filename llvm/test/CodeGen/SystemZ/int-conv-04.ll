; Test zero extensions from a byte to an i64.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test register extension, starting with an i32.
define i64 @f1(i32 %a) {
; CHECK-LABEL: f1:
; CHECK: llgcr %r2, %r2
; CHECK: br %r14
  %byte = trunc i32 %a to i8
  %ext = zext i8 %byte to i64
  ret i64 %ext
}

; ...and again with an i64.
define i64 @f2(i64 %a) {
; CHECK-LABEL: f2:
; CHECK: llgcr %r2, %r2
; CHECK: br %r14
  %byte = trunc i64 %a to i8
  %ext = zext i8 %byte to i64
  ret i64 %ext
}

; Check ANDs that are equivalent to zero extension.
define i64 @f3(i64 %a) {
; CHECK-LABEL: f3:
; CHECK: llgcr %r2, %r2
; CHECK: br %r14
  %ext = and i64 %a, 255
  ret i64 %ext
}

; Check LLGC with no displacement.
define i64 @f4(ptr %src) {
; CHECK-LABEL: f4:
; CHECK: llgc %r2, 0(%r2)
; CHECK: br %r14
  %byte = load i8, ptr %src
  %ext = zext i8 %byte to i64
  ret i64 %ext
}

; Check the high end of the LLGC range.
define i64 @f5(ptr %src) {
; CHECK-LABEL: f5:
; CHECK: llgc %r2, 524287(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524287
  %byte = load i8, ptr %ptr
  %ext = zext i8 %byte to i64
  ret i64 %ext
}

; Check the next byte up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f6(ptr %src) {
; CHECK-LABEL: f6:
; CHECK: agfi %r2, 524288
; CHECK: llgc %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524288
  %byte = load i8, ptr %ptr
  %ext = zext i8 %byte to i64
  ret i64 %ext
}

; Check the high end of the negative LLGC range.
define i64 @f7(ptr %src) {
; CHECK-LABEL: f7:
; CHECK: llgc %r2, -1(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -1
  %byte = load i8, ptr %ptr
  %ext = zext i8 %byte to i64
  ret i64 %ext
}

; Check the low end of the LLGC range.
define i64 @f8(ptr %src) {
; CHECK-LABEL: f8:
; CHECK: llgc %r2, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524288
  %byte = load i8, ptr %ptr
  %ext = zext i8 %byte to i64
  ret i64 %ext
}

; Check the next byte down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f9(ptr %src) {
; CHECK-LABEL: f9:
; CHECK: agfi %r2, -524289
; CHECK: llgc %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524289
  %byte = load i8, ptr %ptr
  %ext = zext i8 %byte to i64
  ret i64 %ext
}

; Check that LLGC allows an index
define i64 @f10(i64 %src, i64 %index) {
; CHECK-LABEL: f10:
; CHECK: llgc %r2, 524287(%r3,%r2)
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to ptr
  %byte = load i8, ptr %ptr
  %ext = zext i8 %byte to i64
  ret i64 %ext
}

; Test a case where we spill the source of at least one LLGCR.  We want
; to use LLGC if possible.
define void @f11(ptr %ptr) {
; CHECK-LABEL: f11:
; CHECK: llgc {{%r[0-9]+}}, 199(%r15)
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

  %trunc0 = trunc i64 %val0 to i8
  %trunc1 = trunc i64 %val1 to i8
  %trunc2 = trunc i64 %val2 to i8
  %trunc3 = trunc i64 %val3 to i8
  %trunc4 = trunc i64 %val4 to i8
  %trunc5 = trunc i64 %val5 to i8
  %trunc6 = trunc i64 %val6 to i8
  %trunc7 = trunc i64 %val7 to i8
  %trunc8 = trunc i64 %val8 to i8
  %trunc9 = trunc i64 %val9 to i8
  %trunc10 = trunc i64 %val10 to i8
  %trunc11 = trunc i64 %val11 to i8
  %trunc12 = trunc i64 %val12 to i8
  %trunc13 = trunc i64 %val13 to i8
  %trunc14 = trunc i64 %val14 to i8
  %trunc15 = trunc i64 %val15 to i8

  %ext0 = zext i8 %trunc0 to i64
  %ext1 = zext i8 %trunc1 to i64
  %ext2 = zext i8 %trunc2 to i64
  %ext3 = zext i8 %trunc3 to i64
  %ext4 = zext i8 %trunc4 to i64
  %ext5 = zext i8 %trunc5 to i64
  %ext6 = zext i8 %trunc6 to i64
  %ext7 = zext i8 %trunc7 to i64
  %ext8 = zext i8 %trunc8 to i64
  %ext9 = zext i8 %trunc9 to i64
  %ext10 = zext i8 %trunc10 to i64
  %ext11 = zext i8 %trunc11 to i64
  %ext12 = zext i8 %trunc12 to i64
  %ext13 = zext i8 %trunc13 to i64
  %ext14 = zext i8 %trunc14 to i64
  %ext15 = zext i8 %trunc15 to i64

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
