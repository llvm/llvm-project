; Test that we can use OI for byte operations that are expressed as i32
; or i64 operations.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Zero extension to 32 bits, negative constant.
define void @f1(i8 *%ptr) {
; CHECK-LABEL: f1:
; CHECK: oi 0(%r2), 254
; CHECK: br %r14
  %val = load i8, i8 *%ptr
  %ext = zext i8 %val to i32
  %or = or i32 %ext, -2
  %trunc = trunc i32 %or to i8
  store i8 %trunc, i8 *%ptr
  ret void
}

; Zero extension to 64 bits, negative constant.
define void @f2(i8 *%ptr) {
; CHECK-LABEL: f2:
; CHECK: oi 0(%r2), 254
; CHECK: br %r14
  %val = load i8, i8 *%ptr
  %ext = zext i8 %val to i64
  %or = or i64 %ext, -2
  %trunc = trunc i64 %or to i8
  store i8 %trunc, i8 *%ptr
  ret void
}

; Zero extension to 32 bits, positive constant.
define void @f3(i8 *%ptr) {
; CHECK-LABEL: f3:
; CHECK: oi 0(%r2), 254
; CHECK: br %r14
  %val = load i8, i8 *%ptr
  %ext = zext i8 %val to i32
  %or = or i32 %ext, 254
  %trunc = trunc i32 %or to i8
  store i8 %trunc, i8 *%ptr
  ret void
}

; Zero extension to 64 bits, positive constant.
define void @f4(i8 *%ptr) {
; CHECK-LABEL: f4:
; CHECK: oi 0(%r2), 254
; CHECK: br %r14
  %val = load i8, i8 *%ptr
  %ext = zext i8 %val to i64
  %or = or i64 %ext, 254
  %trunc = trunc i64 %or to i8
  store i8 %trunc, i8 *%ptr
  ret void
}

; Sign extension to 32 bits, negative constant.
define void @f5(i8 *%ptr) {
; CHECK-LABEL: f5:
; CHECK: oi 0(%r2), 254
; CHECK: br %r14
  %val = load i8, i8 *%ptr
  %ext = sext i8 %val to i32
  %or = or i32 %ext, -2
  %trunc = trunc i32 %or to i8
  store i8 %trunc, i8 *%ptr
  ret void
}

; Sign extension to 64 bits, negative constant.
define void @f6(i8 *%ptr) {
; CHECK-LABEL: f6:
; CHECK: oi 0(%r2), 254
; CHECK: br %r14
  %val = load i8, i8 *%ptr
  %ext = sext i8 %val to i64
  %or = or i64 %ext, -2
  %trunc = trunc i64 %or to i8
  store i8 %trunc, i8 *%ptr
  ret void
}

; Sign extension to 32 bits, positive constant.
define void @f7(i8 *%ptr) {
; CHECK-LABEL: f7:
; CHECK: oi 0(%r2), 254
; CHECK: br %r14
  %val = load i8, i8 *%ptr
  %ext = sext i8 %val to i32
  %or = or i32 %ext, 254
  %trunc = trunc i32 %or to i8
  store i8 %trunc, i8 *%ptr
  ret void
}

; Sign extension to 64 bits, positive constant.
define void @f8(i8 *%ptr) {
; CHECK-LABEL: f8:
; CHECK: oi 0(%r2), 254
; CHECK: br %r14
  %val = load i8, i8 *%ptr
  %ext = sext i8 %val to i64
  %or = or i64 %ext, 254
  %trunc = trunc i64 %or to i8
  store i8 %trunc, i8 *%ptr
  ret void
}
